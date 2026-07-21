import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Official AI Humanizer - Pivot Engine", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HumanizeRequest(BaseModel):
    text: str

# دالة مجانية وسريعة للترجمة عبر محرك Google Translate المباشر
def google_translate(text: str, target_lang: str) -> str:
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_lang}&dt=t&q={requests.utils.quote(text)}"
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            translated = "".join([item[0] for item in data[0] if item[0]])
            return translated
    except Exception:
        pass
    return text  # العودة للنص في حال فشل الاتصال المؤقت

# دالة الاستدعاء المباشر لنموذج DeepSeek عبر OpenRouter
def call_deepseek(messages: list, temp: float, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek/deepseek-chat",  # استدعاء نموذج DeepSeek الرسمي
        "messages": messages,
        "temperature": temp,                # الاعتماد على درجة الحرارة 1.3 الموصى بها في الأداة
        "max_tokens": 2000
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
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is missing on Render.")

    try:
        # ---- الخطوة 1: اعادة صياغة عبر DeepSeek وتحويلها للغة الصينية (ZH) ----
        pass1_messages = [
            {
                "role": "system",
                "content": "You are a multilingual AI text transformation pipeline. Rewrite the input text to preserve its core academic meaning, but express it entirely in Simplified Chinese (ZH) with structural variation."
            },
            {"role": "user", "content": text}
        ]
        chinese_text = call_deepseek(pass1_messages, 1.0, api_key)

        # ---- الخطوة 2: ترجمة النص الصيني إلى اللغة التركية (TR) عبر Google Translate ----
        turkish_text = google_translate(chinese_text, "tr")

        # ---- الخطوة 3: إعادة بناء النص إلى الإنجليزية البشري الصافية عبر DeepSeek (حرارة 1.3) ----
        pass3_messages = [
            {
                "role": "system",
                "content": (
                    "You are a master English reconstruction engine. Reconstruct the provided text back into highly natural, "
                    "fluent, and professional English. Remove all translation artifacts, destroy any lingering robotic sentence flows, "
                    "and deliver clean human-sounding prose. Output ONLY the final reconstructed English text without wrappers."
                )
            },
            {"role": "user", "content": turkish_text}
        ]
        final_english_text = call_deepseek(pass3_messages, 1.3, api_key)

        return {"humanized_text": final_english_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "working", "message": "DeepSeek Multi-Lingual Pivot Humanizer API is live!"}
