import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Official AI Multi-Pass Humanizer Engine", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HumanizeRequest(BaseModel):
    text: str

def call_openrouter_pass(system_prompt: str, user_text: str, temp: float, top_p: float, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        "temperature": temp,
        "top_p": top_p,
        "max_tokens": 2500
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
    if r.status_code == 200:
        return r.json()['choices'][0]['message']['content'].strip()
    else:
        raise HTTPException(status_code=r.status_code, detail=f"DeepSeek API Error: {r.text}")

def run_multipass_humanizer(text: str, api_key: str) -> str:
    # --- التمريرة الأولى: التفكيك المعجمي والتخلص من الكلمات النمطية ---
    pass1_prompt = (
        "You are an expert copyeditor. Rewrite the input text to completely strip away all AI buzzwords, "
        "robotic transition words (like 'Furthermore', 'Moreover', 'Therefore', 'Thus', 'Consequently'), and overly formal academic padding. "
        "Focus on simple, direct human vocabulary. Keep the core facts intact. Output ONLY the rewritten text."
    )
    draft_1 = call_openrouter_pass(pass1_prompt, text, temp=0.75, top_p=0.88, api_key=api_key)

    # --- التمريرة الثانية: حقن التباين النحوي الصارم (Burstiness & Sequence Disruption) ---
    pass2_prompt = (
        "You are a native academic researcher. Take this draft and disrupt its structural patterns to beat AI detectors like GPTZero and Turnitin.\n"
        "1. FORCE HIGH BURSTINESS: Rapidly alternate sentence lengths. Use extremely short, blunt sentences (3-5 words) directly next to longer analytical ones.\n"
        "2. RE-SEQUENCE IDEAS: Do not follow a predictable, linear summary flow. Lead with the core analytical argument or central conflict first.\n"
        "3. DE-NOMINALIZE: Replace passive noun phrases with active, direct verbs.\n"
        "Output ONLY the structurally transformed text."
    )
    draft_2 = call_openrouter_pass(pass2_prompt, draft_1, temp=0.90, top_p=0.92, api_key=api_key)

    # --- التمريرة الثالثة: الصقل العفوي النهائي (Natural Polish) ---
    pass3_prompt = (
        "You are an English professor giving the final human polish to an article. "
        "Ensure the text sounds authentically human, with natural pacing, organic thought transitions, and zero artificial symmetry. "
        "Remove any remaining intro/outro remarks. Output ONLY the raw final humanized prose."
    )
    final_output = call_openrouter_pass(pass3_prompt, draft_2, temp=0.85, top_p=0.90, api_key=api_key)

    return final_output

@app.post("/humanize")
def humanize_text(request: HumanizeRequest):
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY environment variable is missing on Render.")

    try:
        final_text = run_multipass_humanizer(text, api_key)
        return {"humanized_text": final_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "working", "message": "Multi-Pass Pipeline Engine v5.0 is Active!"}
