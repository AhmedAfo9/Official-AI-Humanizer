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
        "model": "meta-llama/llama-3.1-8b-instruct",
        "messages": messages,
        "temperature": temp,  # تثبيت الحرارة لمنع الشطحات والعشوائية
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

    try:
        # ---- المرور الأول: تنظيف المفردات والكلمات الركيكة (الحرارة 0.65) ----
        pass1_messages = [
            {
                "role": "system", 
                "content": (
                    f"You are a native human academic editor. Rewrite the text to eliminate all AI "
                    f"buzzwords, formal fluff, and robotic vocabulary according to these strict rules:\n\n{skill_prompt}\n\n"
                    "Focus on clean, natural, direct phrasing. Output ONLY the raw rewritten text."
                )
            },
            {"role": "user", "content": text}
        ]
        draft_text = call_openrouter(pass1_messages, 0.65, 0.88, api_key)

        # ---- المرور الثاني: تثبيت النمط النحوي الموزون ومنع الروابط الروبوتية (الحرارة 0.68) ----
        pass2_messages = [
            {
                "role": "system", 
                "content": (
                    "You are an academic researcher locking in a human writing voice to guarantee 0% AI detection. "
                    "You must preserve the meaning of the draft while applying this strict, stabilized structural pattern:\n\n"
                    "1. STABILIZED BURSTINESS: Force a steady mix of punchy short sentences (4-7 words) right after long, descriptive ones. Never let three sentences in a row have similar lengths.\n"
                    "2. ZERO DISCOURSE MARKERS: Completely ban transitions like 'Furthermore', 'Moreover', 'Therefore', 'Thus', 'In addition', 'Consequently', or 'In conclusion'. Just advance the thoughts directly.\n"
                    "3. ACTIVE HUMAN CHOICE: Use active voice and direct phrasing ('We analyzed', 'This shows') rather than heavy passive constructs ('An analysis was performed').\n"
                    "4. NO REPETITION: Do not start consecutive sentences with the same subject or noun structure.\n\n"
                    "Output ONLY the final, beautifully stabilized human text."
                )
            },
            {"role": "user", "content": draft_text}
        ]
        final_humanized = call_openrouter(pass2_messages, 0.68, 0.90, api_key)

        return {"humanized_text": final_humanized}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "working", "message": "Stabilized Llama 3.1 Humanizer Core Active."}
