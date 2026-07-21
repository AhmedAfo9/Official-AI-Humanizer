import os
import re
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Official AI Humanizer Engine", version="6.0")

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

def clean_paragraph_structure(original_text: str, humanized_text: str) -> str:
    """
    دالة بايثون برمجية لمنع تقطيع الفقرات بدون الاعتماد على الذكاء الاصطناعي.
    """
    # إذا كان النص الأصلي المكتوب عبارة عن فقرة واحدة متصلة (لا يحتوي على فاصل فقرات \n\n)
    if "\n\n" not in original_text.strip():
        # دمج كل الأسطر المتقطعة في فقرة واحدة متصلة وتعديل المسافات الزائدة
        lines = [line.strip() for line in humanized_text.splitlines() if line.strip()]
        merged_paragraph = " ".join(lines)
        return re.sub(r'\s+', ' ', merged_paragraph)
    else:
        # إذا كان النص الأصلي مقسماً إلى فقرات، يتم تنظيف كل فقرة على حدة دون تقطيع داخلي
        raw_paragraphs = humanized_text.split("\n\n")
        cleaned_paragraphs = []
        for p in raw_paragraphs:
            lines = [line.strip() for line in p.splitlines() if line.strip()]
            if lines:
                cleaned_paragraphs.append(" ".join(lines))
        return "\n\n".join(cleaned_paragraphs)

def run_v5_engine(text: str, api_key: str) -> str:
    # --- التمريرة 1: التنظيف المعجمي وإلغاء الكلمات الأكاديمية الثقيلة ---
    pass1_prompt = (
        "You are an expert copyeditor. Rewrite the input text to completely strip away all AI buzzwords, "
        "robotic transition words (like 'Furthermore', 'Moreover', 'Therefore', 'Thus', 'Consequently'), and overly formal academic padding. "
        "Focus on simple, direct human vocabulary. Output ONLY the rewritten text."
    )
    draft_1 = call_openrouter_pass(pass1_prompt, text, temp=0.75, top_p=0.88, api_key=api_key)

    # --- التمريرة 2: التفكيك الهيكلي العنيف ورفع التباين النحوي (Burstiness) ---
    pass2_prompt = (
        "You are a native academic researcher. Take this draft and disrupt its structural patterns to beat AI detectors like GPTZero and Turnitin.\n"
        "1. FORCE HIGH BURSTINESS: Rapidly alternate sentence lengths. Use extremely short, blunt sentences (3-5 words) directly next to longer analytical ones.\n"
        "2. RE-SEQUENCE IDEAS: Do not follow a predictable summary flow. Lead with the core analytical argument first.\n"
        "3. DE-NOMINALIZE: Replace passive noun phrases with active, direct verbs.\n"
        "Output ONLY the structurally transformed text."
    )
    draft_2 = call_openrouter_pass(pass2_prompt, draft_1, temp=0.90, top_p=0.92, api_key=api_key)

    # --- التمريرة 3: الصقل النهائي العفوي ---
    pass3_prompt = (
        "You are an English professor giving the final human polish to an article. "
        "Ensure the text sounds authentically human, with natural pacing, organic thought transitions, and zero artificial symmetry. "
        "Output ONLY the raw final humanized text."
    )
    raw_output = call_openrouter_pass(pass3_prompt, draft_2, temp=0.85, top_p=0.90, api_key=api_key)

    # تنظيف وتجميع الفقرات برمجياً في الباكيند
    return clean_paragraph_structure(text, raw_output)

@app.post("/humanize")
def humanize_text(request: HumanizeRequest):
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY environment variable is missing on Render.")

    try:
        final_text = run_v5_engine(text, api_key)
        return {"humanized_text": final_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "working", "message": "Engine v6.0 (High Performance + Auto-Clean) Active!"}
