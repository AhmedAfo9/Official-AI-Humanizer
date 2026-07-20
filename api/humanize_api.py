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
        "model": "meta-llama/llama-3.1-8b-instruct",  # تبديل المحرك إلى ملك الأنسنة والتخفي
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

    try:
        # ---- المرحلة الأولى: تفكيك العبارات وتغيير المعجم اللغوي ----
        pass1_messages = [
            {
                "role": "system", 
                "content": (
                    f"You are a native human academic writer. Completely rewrite the input text to convey the exact same core ideas "
                    f"but without using any typical machine phrasing or predictable words. Apply these 33 rules strictly:\n\n{skill_prompt}\n\n"
                    "Write organically. Output ONLY the rewritten text."
                )
            },
            {"role": "user", "content": text}
        ]
        draft_text = call_openrouter(pass1_messages, 0.85, 0.9, api_key)

        # ---- المرحلة الثانية: كسر القوالب النحوية وحظر الروابط الآلية ----
        pass2_messages = [
            {
                "role": "system", 
                "content": (
                    "You are an English professor editing a draft to be 100% indistinguishable from human writing. "
                    "Your primary goal is to bypass advanced statistical AI detectors. Implement these strict structural mandates:\n\n"
                    "1. TOTAL TRANSITION BAN: Never use words like 'Furthermore', 'Moreover', 'Therefore', 'Thus', 'Consequently', or 'In conclusion'. Just state the sentences back-to-back naturally.\n"
                    "2. HIGH BURSTINESS: Vary your sentence structures chaotically. Use a long, descriptive sentence followed immediately by a short, blunt 3-word sentence.\n"
                    "3. HUMAN PACING: Write with a natural, slightly flawed human flow. Avoid perfect robotic symmetry or over-polished transitions.\n\n"
                    "Output ONLY the final raw text."
                )
            },
            {"role": "user", "content": draft_text}
        ]
        final_humanized = call_openrouter(pass2_messages, 0.90, 0.93, api_key)

        return {"humanized_text": final_humanized}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "working", "message": "Llama-Powered Adversarial Core Active."}
