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

    # قراءة القواعد الـ 33
    skill_prompt = ""
    if os.path.exists("SKILL.md"):
        with open("SKILL.md", "r", encoding="utf-8") as f:
            skill_prompt = f.read()
    else:
        skill_prompt = "Apply strict academic humanization rules, removing all AI tells."

    try:
        # ---- المرحلة الأولى: الأنسنة وتطبيق الـ 33 قاعدة لغوياً ----
        pass1_messages = [
            {
                "role": "system", 
                "content": f"You are an expert academic translator and editor. Rewrite the input text to strictly eliminate all AI vocabulary, promotional hedge phrases, and formulaic structures according to these 33 strict rules:\n\n{skill_prompt}\n\nMaintain core semantics but loosen the structure. Output ONLY the raw rewritten text."
            },
            {"role": "user", "content": text}
        ]
        draft_text = call_openrouter(pass1_messages, 0.8, 0.9, api_key)

        # ---- المرحلة الثانية (السر): الفحص والتدقيق العكسي لتدمير البصمة التركيبية ----
        pass2_messages = [
            {
                "role": "system", 
                "content": "You are an adversarial linguistic auditor. Review the provided text to detect and completely eliminate any lingering structural signs of AI writing (such as perfectly symmetrical sentences, predictable rhythmic structures, or robotic transitions). \n\nCRITICAL INSTRUCTION:\n1. Introduce high Burstiness: Vary sentence lengths aggressively (mix very short, punchy sentences with longer, organic ones).\n2. Destroy robotic fluency: Break the slick, overly polished flow. Make it sound like an authentic human academic writing organically from the mind. \n3. Ensure the final text reads naturally in English with zero robotic predictability. Output ONLY the absolute final humanized text."
            },
            {"role": "user", "content": draft_text}
        ]
        final_humanized = call_openrouter(pass2_messages, 0.95, 0.98, api_key)

        return {"humanized_text": final_humanized}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "working", "message": "Official AI Humanizer API is live with Double-Pass Audit Engine!"}
