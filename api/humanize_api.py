import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Official AI Humanizer - Direct Engine", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HumanizeRequest(BaseModel):
    text: str

def call_deepseek_humanizer(text: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # هندسة أوامر هجومية لإلغاء أي بصمة رقمية وتفكيك الجمل
    system_prompt = (
        "You are a native English academic researcher and senior editor. "
        "Your sole task is to rewrite the provided text so that it completely bypasses modern AI detectors (GPTZero, Turnitin) "
        "while maintaining the core academic ideas.\n\n"
        "STRICT REWRITING RULES:\n"
        "1. HIGH BURSTINESS: Create an erratic, organic sentence flow. Mix very short sentences (3-6 words) with longer, descriptive ones.\n"
        "2. BAN AI TRANSITIONS: Absolute ban on 'Furthermore', 'Moreover', 'Therefore', 'Thus', 'Consequently', 'In addition', 'In conclusion'. Never start a sentence with these.\n"
        "3. DE-NOMINALIZE: Change heavy noun phrases into active, direct statements (e.g., replace 'The implementation of X was conducted' with 'We implemented X').\n"
        "4. NO TEXTBOOK SYMMETRY: Avoid perfectly balanced or repetitive sentence setups. Make the rhythm sound like a human writing thoughts directly from memory.\n"
        "5. Output ONLY the raw final humanized text. No disclaimers, no intros."
    )

    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.85,
        "top_p": 0.92,
        "max_tokens": 2500
    }
    
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
    if r.status_code == 200:
        return r.json()['choices'][0]['message']['content'].strip()
    else:
        raise HTTPException(status_code=r.status_code, detail=f"DeepSeek API Error: {r.text}")

@app.post("/humanize")
def humanize_text(request: HumanizeRequest):
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY environment variable is missing on Render.")

    try:
        final_text = call_deepseek_humanizer(text, api_key)
        return {"humanized_text": final_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "working", "message": "Direct DeepSeek Humanizer Engine is Live!"}
