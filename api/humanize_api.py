import os
import re
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Official AI Sovereign Anti-Detection Core", version="10.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HumanizeRequest(BaseModel):
    text: str

def python_deep_cleanse(text: str) -> str:
    """
    تطهير حتمي مباشر في البايثون لأكثر العبارات التي ترصدها كواشف Turnitin و GPTZero
    """
    patterns = {
        r'\bplays a (crucial|vital|pivotal|key) role in\b': 'drives',
        r'\bit is (important|worth) noting that\b': 'notably,',
        r'\bserves as a testament to\b': 'demonstrates',
        r'\bdelves into\b': 'explores',
        r'\bin order to\b': 'to',
        r'\bfurthermore,?\b': 'also,',
        r'\bmoreover,?\b': 'beyond this,',
        r'\bconsequently,?\b': 'thus,',
        r'\btherefore,?\b': 'hence,',
        r'\bin conclusion,?\b': 'ultimately,',
        r'\bmultifaceted\b': 'varied',
        r'\bintricate\b': 'complex',
        r'\bshed light on\b': 'clarify',
    }
    cleaned = text
    for pat, repl in patterns.items():
        cleaned = re.sub(pat, repl, cleaned, flags=re.IGNORECASE)
    return cleaned

def execute_sovereign_transform(block_text: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # توجيهات ألسنية صارمة لتفكيك النمط الإحصائي المألوف
    system_prompt = (
        "You are an elite linguistic strategist tasked with completely destroying AI detection signatures (GPTZero, Turnitin) "
        "while preserving the original academic core facts.\n\n"
        "UNCOMPROMISING REWRITING RULES:\n"
        "1. EXTREME SYNTACTIC VARIATION: Invert standard sentence order. Begin sentences with adverbial clauses, prepositional phrases, or direct action verbs.\n"
        "2. UNPREDICTABLE RHYTHM: Alternate abruptly between punchy short sentences (4-7 words) and winding analytical thoughts.\n"
        "3. DE-NOMINALIZE: Convert all passive noun-heavy phrasing into active, vivid verbs.\n"
        "4. ABSOLUTE BAN ON AI FORMULAS: Never use robotic academic connectors or standard AI buzzwords.\n"
        "5. PRESERVE BLOCK LAYOUT: Do not introduce line breaks or markdown formatting. Output ONLY the rewritten text of this paragraph."
    )

    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": block_text}
        ],
        "temperature": 0.95,
        "top_p": 0.93,
        "frequency_penalty": 0.60,  # عقوبة قاسية لمنع تكرار أنماط الجمل
        "presence_penalty": 0.40,
        "max_tokens": 2000
    }
    
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
    if r.status_code == 200:
        res = r.json()['choices'][0]['message']['content'].strip()
        res = re.sub(r'\s*\n\s*', ' ', res)
        return python_deep_cleanse(res)
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
        
        humanized = execute_sovereign_transform(block_str, api_key)
        processed_blocks.append(humanized)
        
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
    return {"status": "working", "message": "Sovereign Anti-Detection Core v10.0 Active!"}
