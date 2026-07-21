import os
import re
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Official AI High-Entropy Humanizer", version="8.5")

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
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_prompt = (
        "You are a native English scholar and editorial strategist. "
        "Your task is to completely rewrite the input text to achieve a 0% AI detection score on GPTZero and Turnitin, "
        "while strictly preserving the original core ideas and academic meaning.\n\n"
        "HUMANIZATION TACTICS:\n"
        "1. HIGH SYNTACTIC DIVERSITY: Mix sentence structures continuously. Place complex clauses right next to short, direct statements.\n"
        "2. ACTIVE & DIRECT PHRASING: Convert heavy passive noun structures into clean, active human verbs.\n"
        "3. BAN AI MARKERS: Zero tolerance for standard transition words (Furthermore, Moreover, Consequently, In addition, Therefore, Thus, In summary).\n"
        "4. NATURAL CADENCE: Write with organic human cadence, varied vocabulary, and direct thought flow.\n"
        "5. PRESERVE BLOCK INTEGRITY: Do NOT split paragraphs, add bullet points, or output conversational filler. Return ONLY the rewritten text of this paragraph."
    )

    payload = {
        # التبديل إلى Llama 3.3 70B لتدمير التوقع الإحصائي للمفردات
        "model": "meta-llama/llama-3.3-70b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": block_text}
        ],
        "temperature": 0.95,
        "top_p": 0.92,
        "frequency_penalty": 0.45,
        "presence_penalty": 0.35,
        "max_tokens": 2000
    }
    
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
    if r.status_code == 200:
        res = r.json()['choices'][0]['message']['content'].strip()
        res = re.sub(r'\s*\n\s*', ' ', res)
        return res
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
    return {"status": "working", "message": "High-Entropy Llama-3.3 Humanizer v8.5 is Active!"}
