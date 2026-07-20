import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI(title="Official AI Humanizer API", version="1.0")

# تفعيل الـ CORS بشكل صحيح تماماً لاستقبال الطلبات من صفحة Cloudflare
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
    
    # قراءة القواعد الـ 33 من ملف القواعد المحلي
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
                "content": f"You are an expert text humanizer implementing the strict Wikipedia AI cleanup rules outlined below:\n\n{skill_prompt}\n\nCRITICAL RULE: Meticulously maintain the exact original paragraphing, spacing, and line-break structure of the source text. Output ONLY the final humanized text cleanly. Do not add any intros, chatbot meta-commentary, or markdown wrappers."
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
        "max_tokens": 2000  # قيد ذكي لتجاوز فحص الرصيد الأقصى في OpenRouter
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
