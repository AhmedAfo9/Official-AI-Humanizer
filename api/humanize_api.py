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

    skill_prompt = ""
    if os.path.exists("SKILL.md"):
        with open("SKILL.md", "r", encoding="utf-8") as f:
            skill_prompt = f.read()
    else:
        skill_prompt = "Apply strict academic humanization rules."

    try:
        # ---- المرحلة الأولى: تطهير المفردات والألفاظ بناءً على الـ 33 قاعدة ----
        pass1_messages = [
            {
                "role": "system", 
                "content": f"Rewrite the following text to eliminate all textbook AI vocabulary and promotional language using these rules:\n\n{skill_prompt}\n\nFocus on lexical substitution and raw meaning extraction. Output ONLY the rewritten prose."
            },
            {"role": "user", "content": text}
        ]
        draft_text = call_openrouter(pass1_messages, 0.85, 0.92, api_key)

        # ---- المرحلة الثانية والأقوى: سحق الـ 40% المتبقية (تدمير الهيكل المتوقع والتون المتناسق) ----
        pass2_messages = [
            {
                "role": "system", 
                "content": (
                    "You are a ruthless native English academic editor combating AI detection algorithms (GPTZero/Turnitin). "
                    "Your goal is to destroy the remaining 40% structural predictability in the text. "
                    "Apply these absolute structural overrides:\n\n"
                    "1. STRICT TRANSITION BAN: Completely ban mechanical transition words like 'Furthermore', 'Moreover', 'In addition', 'Consequently', 'Therefore', 'Thus', 'Hence'. Connect ideas by overlapping concepts or using direct narrative progression instead.\n"
                    "2. DE-NOMINALIZATION: Convert heavy robotic noun phrases into active verbal clause structures (e.g., instead of 'the utilization of X provides an enhancement to Y', rewrite to 'using X improves Y').\n"
                    "3. ASYMMETRICAL SYNTAX: Break any uniform pattern of subject-verb-object. Start some sentences with prepositional phrases, some with short assertions, and others with complex dependencies. Create an erratic, organic human text flow.\n"
                    "4. ACADEMIC IDIOMS: Inject natural, idiomatic academic phrasing that native researchers use naturally but LLMs rarely generate due to low statistical probability.\n\n"
                    "Output ONLY the final, polished human prose. No meta-commentary, no introductory text."
                )
            },
            {"role": "user", "content": draft_text}
        ]
        final_humanized = call_openrouter(pass2_messages, 0.98, 0.99, api_key)

        return {"humanized_text": final_humanized}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "working", "message": "Official AI Humanizer API is live with Ultra-Variance Engine!"}
