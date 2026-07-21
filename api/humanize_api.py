import os
import re
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Official AI Hybrid Purge Engine", version="9.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HumanizeRequest(BaseModel):
    text: str

def python_ngram_purge(text: str) -> str:
    """
    مرشح بايثون حتمي لتطهير النص من عبارات البصمة الرقمية الشائعة للذكاء الاصطناعي.
    """
    replacements = {
        r'\bplays a crucial role in\b': 'is key to',
        r'\bplays a vital role in\b': 'matters significantly for',
        r'\bit is important to note that\b': 'notably,',
        r'\bit is worth noting that\b': 'importantly,',
        r'\bdelves into\b': 'examines',
        r'\bserves as a testament to\b': 'shows',
        r'\bin order to\b': 'to',
        r'\bfurthermore,?\b': 'also,',
        r'\bmoreover,?\b': 'in addition,',
        r'\bconsequently,?\b': 'as a result,',
        r'\btherefore,?\b': 'thus,',
        r'\bin conclusion,?\b': 'overall,',
        r'\bmultifaceted\b': 'complex',
        r'\bvital component\b': 'main part',
        r'\bshed light on\b': 'explain',
    }
    
    purged_text = text
    for pattern, replacement in replacements.items():
        purged_text = re.sub(pattern, replacement, purged_text, flags=re.IGNORECASE)
        
    return purged_text

def humanize_single_block(block_text: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # أوامر هندسية لتغيير بناء الجملة وتقديم ظروف الزمان والمكان
    system_prompt = (
        "You are an academic copyeditor rewriting a text to be indistinguishable from human writing.\n\n"
        "REWRITING DIRECTIVES:\n"
        "1. CLAUSE REORDERING: Frequently start sentences with prepositional phrases, dependent clauses, or main verbs rather than subject-first templates.\n"
        "2. VARY SENTENCE RHYTHM: Create a sharp contrast between brief sentences (4-7 words) and longer analytical ones.\n"
        "3. DE-NOMINALIZE: Change passive abstract nouns into direct active human actions.\n"
        "4. PRESERVE BLOCK INTEGRITY: Do not add extra line breaks or formatting. Return ONLY the transformed text of this exact block."
    )

    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": block_text}
        ],
        "temperature": 0.92,
        "top_p": 0.90,
        "frequency_penalty": 0.50,
        "presence_penalty": 0.30,
        "max_tokens": 2000
    }
    
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
    if r.status_code == 200:
        res = r.json()['choices'][0]['message']['content'].strip()
        res = re.sub(r'\s*\n\s*', ' ', res)
        # تطبيق مرشح البايثون الحتمي بعد خروج النص من النموذج
        return python_ngram_purge(res)
    else:
        raise HTTPException(status_code=r.status_code, detail=f"OpenRouter API Error: {r.text}")

def process_text_structure(full_text: str, api_key: str) -> str:
    normalized = full_text.replace('\r\n', '\n')
    blocks = normalized.split('\n\n')
    
    processed_blocks = []
    for block in blocks:
        block_str = block.strip()
        if not block_str:
            continue
        
        humanized_block = humanize_single_block(block_str, api_key)
        processed_blocks.append(humanized_block)
        
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
    return {"status": "working", "message": "Hybrid N-Gram Purge Engine v9.0 is Active!"}
