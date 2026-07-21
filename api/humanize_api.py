import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Official AI Cohesive Humanizer Engine", version="5.5")

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

def run_cohesive_humanizer(text: str, api_key: str) -> str:
    # --- المرحلة الأولى: التفكيك وإلغاء القوالب الجاهزة مع الحفاظ على الفقرة ---
    pass1_prompt = (
        "You are an expert academic editor. Rewrite the text to replace predictable AI transitions "
        "(e.g., 'Furthermore', 'Moreover', 'In addition', 'Consequently') with direct, active human phrasing.\n"
        "CRITICAL FORMATTING RULE: Maintain the exact same paragraph structure as the input. "
        "Do NOT split single paragraphs into multiple fragmented blocks or lines."
    )
    draft_1 = call_openrouter_pass(pass1_prompt, text, temp=0.78, top_p=0.88, api_key=api_key)

    # --- المرحلة الثانية: الأنسنة النقدية وحقن العفوية اللغوية ---
    pass2_prompt = (
        "You are a native English scholar rewriting a draft to achieve a low score on AI detectors (GPTZero, Turnitin).\n\n"
        "RULES FOR NATURAL FLOW:\n"
        "1. PARAGRAPH COHESION: Output the text as cohesive continuous paragraphs matching the original input layout. Never insert artificial line breaks.\n"
        "2. VARIED SYNTAX WITHIN PARAGRAPHS: Alternate short punchy sentences with compound analytical ones naturally inside the same paragraph.\n"
        "3. CRITICAL PERSPECTIVE: Shift dry summary statements into active critical evaluation.\n"
        "4. NO AI TRANSITIONS: Ban all robotic connective markers.\n\n"
        "Output ONLY the final humanized text."
    )
    final_output = call_openrouter_pass(pass2_prompt, draft_1, temp=0.86, top_p=0.92, api_key=api_key)

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
        final_text = run_cohesive_humanizer(text, api_key)
        return {"humanized_text": final_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "working", "message": "Cohesive Humanizer Core v5.5 Active!"}
