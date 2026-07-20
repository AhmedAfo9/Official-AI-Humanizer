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

@app.post("/humanize")
def humanize_text(request: HumanizeRequest):
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    skill_prompt = ""
    if os.path.exists("SKILL.md"):
        with open("SKILL.md", "r", encoding="utf-8") as f:
            skill_prompt = f.read()
    else:
        skill_prompt = "Apply strict academic humanization rules, removing all AI tells."

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY environment variable is missing on Render.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {
                "role": "system", 
                "content": f"You are a master of advanced stylistic linguistic humanization. Your job is to completely rewrite the input text to completely eliminate academic AI structural signatures, bypassing advanced detectors like GPTZero and Turnitin while preserving the exact core meaning.\n\nHere are your strict 33 guiding rules:\n\n{skill_prompt}\n\nCRITICAL STRATEGY FOR HUMANIZATION:\n1. Destructure and re-architect the sentence syntax completely. Do not just swap words for synonyms. Change passive voice to active, split monotonous sentences, and introduce dynamic phrase lengths (high burstiness).\n2. Eradicate perfect robotic symmetry. Use natural transition flows. \n3. Keep the original semantic concepts intact, but frame them with human perplexity. Output ONLY the resulting raw text without wrappers or introductions."
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.85,  # رفع الحرارة لزيادة التنوع البشري العشوائي وغير المتوقع
        "top_p": 0.95,        # إجبار النموذج على استخدام خيارات مفردات أكثر جرأة وعمقاً
        "max_tokens": 2000
    }
    
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            result = r.json()['choices'][0]['message']['content'].strip()
            return {"humanized_text": result}
        else:
            raise HTTPException(status_code=r.status_code, detail=f"API Error: {r.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection Failed: {str(e)}")

@app.get("/")
def root():
    return {"status": "working", "message": "Official AI Humanizer API is live!"}
