import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI(title="Official AI Humanizer API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HumanizeRequest(BaseModel):
    text: str

def call_openrouter(messages, temp, top_p, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": messages,
        "temperature": temp,
        "top_p": top_p,
        "max_tokens": 2000
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
    if r.status_code == 200:
        return r.json()['choices'][0]['message']['content'].strip()
    else:
        raise HTTPException(status_code=r.status_code, detail=f"OpenRouter Error: {r.text}")

@app.post("/humanize")
def humanize_text(request: HumanizeRequest):
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is missing.")

    try:
        # ---- المرحلة الأولى: تفكيك النص وتطهير العبارات المكشوفة لغوياً ----
        pass1_messages = [
            {
                "role": "system", 
                "content": (
                    "You are an expert academic editor. Your job is to extract the raw arguments from the input text "
                    "and completely draft them from scratch. Banish all predictable AI vocabulary (pivotal, crucial, delve, furthermore, testament, landscape). "
                    "Write it in a clean, direct, non-robotic academic tone. Output ONLY the raw draft."
                )
            },
            {"role": "user", "content": text}
        ]
        draft_text = call_openrouter(pass1_messages, 0.82, 0.90, api_key)

        # ---- المرحلة الثانية (تدمير البصمة الهيكلية): حقن التنوع النحوي الصارم ----
        pass2_messages = [
            {
                "role": "system", 
                "content": (
                    "You are a native English professor editing a research paper to ensure it has zero robotic predictability (0% AI detection). "
                    "The current draft is grammatically clean but its structural rhythm might trigger algorithms. You MUST bypass them using these strict criteria:\n\n"
                    "1. MAXIMUM BURSTINESS: Force erratic sentence lengths. Alternate a highly complex sentence containing dependent clauses with a very short, blunt, punchy statement (e.g., 'This matters.', 'The data agreed.').\n"
                    "2. ZERO ROBOTIC TRANSITIONS: Completely ban logical markers like 'Therefore', 'Moreover', 'Consequently', 'In addition', 'Thus', 'On one hand'. Instead, create organic transitions where the thoughts link naturally without a formal announcement.\n"
                    "3. SYNTACTIC VARIATION: Avoid repetitive sentence setups. Do not start sentences with the same subject-verb format. Start with prepositional phrases, gerunds, or conditional structures.\n"
                    "4. HUMAN IDIOMS & PHRASING: Frame assertions using active, direct human phrasing (e.g., replace 'A study was conducted by the authors' with 'We ran a study').\n\n"
                    "Output ONLY the final, beautifully asymmetric human prose. Do not introduce it."
                )
            },
            {"role": "user", "content": draft_text}
        ]
        final_humanized = call_openrouter(pass2_messages, 0.88, 0.95, api_key)

        return {"humanized_text": final_humanized}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "working", "message": "Ultra-Adversarial Humanizer Core Active."}
