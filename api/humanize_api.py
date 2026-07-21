import os
import re
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Official AI Block-Preserving Humanizer", version="8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HumanizeRequest(BaseModel):
    text: str

def humanize_single_block(block_text: str, api_key: str) -> str:
    """
    معالجة كتلة نصية واحدة (فقرة أو عنوان) بشكل مستقل لحفظ الهيكل.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # أوامر هندسية قائمة على تفكيك البنية النحوية لتدمير البصمة الإحصائية
    system_prompt = (
        "You are an expert linguistic editor. Your sole task is to rewrite the input paragraph to completely bypass AI detection (GPTZero, Turnitin) "
        "by breaking standard LLM token probability distributions.\n\n"
        "STRICT REWRITING DIRECTIVES:\n"
        "1. CLAUSE INVERSION: Flip sentence structures (e.g., place dependent or adverbial clauses at the start of sentences).\n"
        "2. DE-NOMINALIZE: Convert heavy noun phrases into active, direct verbs.\n"
        "3. NO ROBOTIC TRANSITIONS: Absolute ban on 'Furthermore', 'Moreover', 'Consequently', 'In addition', 'Therefore', 'Thus'.\n"
        "4. SYNTACTIC DIVERSITY: Rapidly alternate between short blunt statements and longer analytical clauses.\n"
        "5. PRESERVE BLOCK INTEGRITY: Do NOT add extra line breaks, bullet points, or commentary inside this paragraph. Return ONLY the rewritten text of this block."
    )

    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": block_text}
        ],
        "temperature": 0.88,
        "top_p": 0.92,
        "max_tokens": 2000
    }
    
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
    if r.status_code == 200:
        res = r.json()['choices'][0]['message']['content'].strip()
        # إلغاء أي أسطر جديدة قد يفتعلها النموذج داخل الفقرة الواحدة
        res = re.sub(r'\s*\n\s*', ' ', res)
        return res
    else:
        raise HTTPException(status_code=r.status_code, detail=f"DeepSeek API Error: {r.text}")

def process_text_structure(full_text: str, api_key: str) -> str:
    """
    تفكيك النص إلى كتل بحسب الفقرات الأصلية وإعادة تجميعها بنفس الترتيب تماماً.
    """
    normalized = full_text.replace('\r\n', '\n')
    # التقسيم بناءً على أسطر الفقرات المزدوجة (\n\n)
    blocks = normalized.split('\n\n')
    
    processed_blocks = []
    for block in blocks:
        block_str = block.strip()
        if not block_str:
            continue
        
        # معالجة الكتلة بمفردها
        humanized_block = humanize_single_block(block_str, api_key)
        processed_blocks.append(humanized_block)
        
    # إعادة التجميع بنفس الفواصل الأصلية المزدوجة
    return "\n\n".join(processed_blocks)

@app.post("/humanize")
def humanize_text(request: HumanizeRequest):
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY environment variable is missing on Render.")

    try:
        final_text = process_text_structure(text, api_key)
        return {"humanized_text": final_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "working", "message": "Block-Structure Humanizer v8.0 is Active!"}
