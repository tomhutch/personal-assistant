import requests
import json
import re

from config import SLM_URL

def parse_intent(transcript, dynamic_system_prompt):
    """Sends transcript to Qwen SLM and parses structured JSON response."""
    print("⚡ Parsing intent via Qwen 3B...")
    payload = {
        "model": "qwen2.5-3b",
        "messages": [
            {"role": "system", "content": dynamic_system_prompt},
            {"role": "user", "content": transcript}
        ],
        "temperature": 0.0,
        "max_tokens": 150
    }

    try:
        slm_resp = requests.post(SLM_URL, json=payload)
        slm_data = slm_resp.json()
        raw_content = slm_data["choices"][0]["message"]["content"].strip()
        
        cleaned_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.MULTILINE).strip()
        return json.loads(cleaned_json)
    except Exception as e:
        print(f"❌ Failed to parse SLM response: {e}")
        return None
