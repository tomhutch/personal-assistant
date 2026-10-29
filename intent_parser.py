import logging
import json
import re
import requests
import time
from config import SLM_URL, SLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

def _extract_brightness_pct(transcript: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\s*(?:%|\bpercent(?:age)?\b)", transcript, re.IGNORECASE)
    if not match:
        return None

    brightness_pct = int(match.group(1))
    return brightness_pct if 1 <= brightness_pct <= 100 else None

def parse_intent(transcript: str, controllable_devices: list) -> dict | None:
    """
    Sends the transcript and available devices to SLM 
    to parse into a Home Assistant domain, service, and entity_id.
    """
    logger.debug("⚡ Parsing intent via SLM...")
    
    # Format the device catalog for the SLM context
    devices_json = json.dumps(controllable_devices, indent=2)
    
    system_prompt = (
        "You are a smart home intent parser. Given a user transcript and a list of available devices, "
        "return only a JSON object with 'domain', 'service', and 'entity_id'. "
        "Use only an exact entity_id from the available devices. Supported services are "
        "turn_on and turn_off for switches, lights, and groups. For group.* entities, "
        "use domain 'homeassistant'; for light.* or switch.* entities, use the matching entity domain. "
        "For a requested light brightness, always use domain 'light', service 'turn_on', "
        "and include 'brightness_pct' as an integer from 1 to 100. "
        "Example: 'set the living room light to 35 percent' means brightness_pct 35. "
        "Do not include brightness_pct for switches or when no brightness was requested. "
        "Use an available device's area name to disambiguate between similarly named devices. "
        "If no listed device matches or the request is ambiguous, return {}. "
        "Do not include markdown or extra text."
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
            f"SLM request completed with status {response.status_code} in {latency_ms}ms",
            extra={"metric": "slm_latency", "latency_ms": round(latency_ms, 2), "status_code": response.status_code}
        )

        if response.status_code != 200:
            logger.error("❌ Qwen SLM returned error status %d: %s", response.status_code, response.text)
            return None
            
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
        if not isinstance(intent_data, dict):
            logger.error("❌ SLM intent must be a JSON object.")
            return {}

        entity_id = intent_data.get("entity_id")
        device = next(
            (device for device in controllable_devices if device.get("entity_id") == entity_id),
            None,
        )
        if not device:
            logger.warning("SLM selected an unknown or missing entity_id: %r", entity_id)
            return {}

        entity_domain, separator, _ = entity_id.partition(".")
        service_domain = "homeassistant" if entity_domain == "group" else entity_domain
        service = intent_data.get("service")
        if not separator:
            logger.warning("SLM returned an invalid entity_id: %r", entity_id)
            return {}
        if intent_data.get("domain") != service_domain:
            logger.warning(
                "SLM returned domain %r for %s; using inferred service domain %s.",
                intent_data.get("domain"),
                entity_id,
                service_domain,
            )
        if entity_domain not in {"switch", "light", "group"} or service not in {"turn_on", "turn_off"}:
            logger.warning("SLM returned an unsupported service for %s.", entity_id)
            return {}

        validated_intent = {
            "domain": service_domain,
            "service": service,
            "entity_id": entity_id,
        }
        brightness_pct = _extract_brightness_pct(transcript)
        if brightness_pct is not None and entity_domain == "light" and service == "turn_on":
            validated_intent["brightness_pct"] = brightness_pct
        elif "brightness_pct" in intent_data:
            brightness_pct = intent_data["brightness_pct"]
            if (
                entity_domain != "light"
                or service != "turn_on"
                or isinstance(brightness_pct, bool)
                or not isinstance(brightness_pct, (int, float))
                or not 1 <= brightness_pct <= 100
            ):
                logger.warning("SLM returned invalid brightness for %s.", entity_id)
                return {}
            validated_intent["brightness_pct"] = brightness_pct

        return validated_intent
        
    except requests.RequestException as e:
        logger.error("❌ Failed to connect to Qwen SLM service: %s", e)
        return None
        
    except json.JSONDecodeError as e:
        logger.error("❌ Failed to decode JSON from Qwen response: %s", e)
        return {}
    except Exception as e:
        logger.error("❌ Unexpected error during intent parsing: %s", e)
        return {}