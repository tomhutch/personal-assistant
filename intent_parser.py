import logging
import json
import requests
import time
from config import SLM_URL, SLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

def parse_intent(transcript: str, controllable_devices: list) -> dict:
    """
    Sends the transcript and available devices to the local Qwen SLM 
    to parse into a Home Assistant domain, service, and entity_id.
    """
    logger.debug("⚡ Parsing intent via SLM...")
    
    # Format the device catalog for the SLM context
    devices_json = json.dumps(controllable_devices, indent=2)
    
    system_prompt = (
        "You are a smart home intent parser. Given a user transcript and a list of available devices, "
        "output a strict JSON object with keys: 'domain', 'service', and 'entity_id'. "
        "Example: {\"domain\": \"switch\", \"service\": \"turn_off\", \"entity_id\": \"switch.plug_1\"}. "
        "Do not include markdown blocks or extra text, just raw JSON."
    )
    
    user_prompt = f"Transcript: \"{transcript}\"\nAvailable Devices:\n{devices_json}"
    
    payload = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    try:
        start_time = time.perf_counter()
        response = requests.post(SLM_URL, json=payload, timeout=SLM_TIMEOUT_SECONDS)
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "SLM request metrics",
            extra={"metric": "slm_latency", "latency_ms": round(latency_ms, 2), "status_code": response.status_code}
        )

        if response.status_code != 200:
            logger.error("❌ Qwen SLM returned error status %d: %s", response.status_code, response.text)
            return {}
            
        data = response.json()
        
        # Robustly extract content from OpenAI-compatible choices structure
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content", "{}")
        else:
            logger.error("❌ Failed to parse SLM response: missing 'choices' key in payload: %s", data)
            return {}
            
        # Clean up potential markdown formatting if the SLM wrapped it
        cleaned_content = content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3]
        cleaned_content = cleaned_content.strip()
        
        intent_data = json.loads(cleaned_content)
        return intent_data
        
    except requests.RequestException as e:
        logger.error("❌ Failed to connect to Qwen SLM service: %s", e)
        return {}
        
    except json.JSONDecodeError as e:
        logger.error("❌ Failed to decode JSON from Qwen response: %s", e)
        return {}
    except Exception as e:
        logger.error("❌ Unexpected error during intent parsing: %s", e)
        return {}