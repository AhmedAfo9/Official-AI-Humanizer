import os
import re
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Official AI Humanizer - Benchmark Engine v7.0", version="7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HumanizeRequest(BaseModel):
    text: str

def clean_and_merge_paragraphs(original_text: str, generated_text: str) -> str:
    """
    دالة بايثون حتمية تمنع تقطيع الفقرات نهائياً وتضمن خروج النص بنفس بيكلية الفقرات الأصلية.
    """
    # تنظيف أي علامات غريبة أو رموز زوائد
    clean_gen = re.sub(r'```.*?\n', '', generated_text)
    clean_gen = clean_gen.replace('```', '').strip()
    
    # إذا كان النص المدخل فقرة واحدة متصلة
    if "\n\n" not in original_text.strip():
        lines = [line.strip() for line in clean_gen.splitlines() if line.strip()]
        merged = " ".join(lines)
        return re.sub(r'\s+', ' ', merged)
    
    # إذا كان النص الأصلي متعدّد الفقرات
    orig_paragraphs = [p.strip() for p in original_text.split("\n\n") if p.strip()]
    gen_paragraphs = [p.strip() for p in clean_gen.split("\n\n") if p.strip()]
    
    # إذا قام النموذج بتقطيع الفقرات زيادة عن اللزوم، نعيد تجميعها برمجياً
    if len(gen_paragraphs) != len(orig_paragraphs) and len(orig_paragraphs) > 0:
        lines = [line.strip() for line in clean_gen.splitlines() if line.strip()]
        full_text = " ".join(lines)
        return re.sub(r'\s+', ' ', full_text)
        
    return clean_gen

def call_human_benchmark_deepseek(text: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # الـ Prompt الجديد المعتمد على عيّنات الأطروحة البشرية الحقيقية (2020)
    system_prompt = (
        "You are an expert human academic writer and linguist. Your goal is to rewrite the input text "
        "so that its style, flow, and vocabulary match genuine human Master's thesis writing (from 2020).\n\n"
        "HUMAN BASELINE EXEMPLAR (Reference Style):\n"
        "\"Analysing a text with a deep comprehensive understanding is the focus of pragmatics. "
        "The main focus is on the communicative part of language which is conceived as an intentional human action. "
        "Pragmatics is concerned with the study of texts whether written or spoken in terms of the context and the functions of its producers.\"\n\n"
        "MANDATORY HUMANIZING RULES:\n"
        "1. PARAGRAPH CONTINUITY: Write in continuous, well-connected paragraphs. Do NOT insert extra line breaks, bullet points, or sentence splits.\n"
        "2. NATURAL CADENCE: Use a natural human flow with varied sentence lengths (12 to 25 words). Avoid forcing artificial ultra-short sentences.\n"
        "3. STRIP AI TRANSITIONS: Do NOT use robotic connectors like 'Furthermore', 'Moreover', 'Consequently', 'In conclusion', or 'In summary'. Use direct conceptual flow.\n"
        "4. SIMPLE DIRECT ACADEMIC VOICE: Use straightforward, clean scholarly language. Avoid fluffy metaphors or exaggerated vocabulary.\n\n"
        "OUTPUT REQUIREMENT: Return ONLY the final rewritten text. No wrappers, no introductions."
    )

    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.82,
        "top_p": 0.90,
        "max_tokens": 2500
    }
    
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
    if r.status_code == 200:
        raw_result = r.json()['choices'][0]['message']['content'].strip()
        return clean_and_merge_paragraphs(text, raw_result)
    else:
        raise HTTPException(status_code=r.status_code, detail=f"DeepSeek API Error: {r.text}")

@app.post("/humanize")
def humanize_text(request: HumanizeRequest):
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY environment variable is missing on Render.")

    try:
        final_text = call_human_benchmark_deepseek(text, api_key)
        return {"humanized_text": final_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "working", "message": "Human Benchmark Engine v7.0 is Live!"}
