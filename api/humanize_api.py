import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Official AI Universal Humanizer Engine", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HumanizeRequest(BaseModel):
    text: str

def call_universal_deepseek_humanizer(text: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # النظام الشامل المضاد لجميع أنواع الكواشف (لكافة أنواع النصوص)
    system_prompt = (
        "You are an elite academic rewriting engine engineered to pass all modern AI detectors (GPTZero, Turnitin, CopyLeaks) with a 0% AI score.\n"
        "Your task is to completely transform the input text—regardless of its topic, format, or genre (summaries, essays, technical reports, literature)—into authentically human-written prose.\n\n"
        "UNIVERSAL REWRITING DIRECTIVES:\n"
        "1. EXTREME BURSTINESS: Strictly force an unpredictable sentence length rhythm. Intersperse blunt, punchy 3-5 word sentences between longer, winding analytical clauses.\n"
        "2. ABSOLUTE TRANSITION BAN: Zero tolerance for AI marker transitions. Never use 'Furthermore', 'Moreover', 'Therefore', 'Thus', 'Consequently', 'In conclusion', 'Soon after', or 'Throughout the text'.\n"
        "3. DE-NOMINALIZE & ACTIVE VOICE: Shift passive and robotic noun clusters into direct, active human verbs (e.g., convert 'An investigation was conducted' into 'We investigated').\n"
        "4. ANTI-CLICHÉ RE-STRUCTURING: If the text is a summary or well-known topic, disrupt its linear progression. Lead with critical arguments or core themes first, then integrate background details organically.\n"
        "5. NATURAL IMPERFECTION: Write with organic pacing that reflects direct human thought progression rather than perfectly balanced robotic symmetry.\n\n"
        "OUTPUT REQUIREMENT: Output ONLY the raw final humanized text. Do NOT wrap in quotes, add intros, or include meta commentary."
    )

    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.88,
        "top_p": 0.94,
        "max_tokens": 3000
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
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY environment variable is missing on Render.")

    try:
        final_text = call_universal_deepseek_humanizer(text, api_key)
        return {"humanized_text": final_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "working", "message": "Universal 0% AI Humanizer Engine is Active!"}
