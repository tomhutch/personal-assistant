import logging
import re
from typing import Any

from hassil import Intents
from hassil.intents import TextSlotList
from hassil.recognize import recognize

logger = logging.getLogger(__name__)


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _extract_brightness_pct(transcript: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\s*(?:%|percent(?:age)?)\b", transcript, re.IGNORECASE)
    if not match:
        return None

    brightness_pct = int(match.group(1))
    return brightness_pct if 1 <= brightness_pct <= 100 else None


def _build_slot_lists(controllable_devices: list) -> dict[str, TextSlotList]:
    name_values = []
    area_values = []

    for device in controllable_devices:
        entity_id = device.get("entity_id") or ""
        name = device.get("name")
        area = device.get("area")

        if name and name not in name_values:
            name_values.append(name)

        entity_name = entity_id.split(".")[-1].replace("_", " ") if entity_id else ""
        if entity_name and entity_name not in name_values:
            name_values.append(entity_name)

        if area and area not in area_values:
            area_values.append(area)

    return {
        "name": TextSlotList.from_strings(sorted(name_values, key=len, reverse=True)),
        "area": TextSlotList.from_strings(sorted(area_values, key=len, reverse=True)),
    }


def _extract_area_hint(transcript: str) -> str | None:
    matches = re.findall(r"\bin\s+(?:the\s+)?([a-z0-9][a-z0-9\s-]+?)(?:[.!?]|$)", transcript.lower())
    if not matches:
        return None
    return matches[-1].strip()


def _find_matching_device(transcript: str, controllable_devices: list, matched_name: str | None) -> dict | None:
    if not matched_name:
        return None

    normalized_target = _normalize_value(matched_name)
    area_hint = _extract_area_hint(transcript)

    candidate_devices = []
    for device in controllable_devices:
        aliases = []
        if device.get("name"):
            aliases.append(device["name"])
        entity_id = device.get("entity_id") or ""
        if entity_id:
            aliases.append(entity_id.split(".")[-1].replace("_", " "))
        if device.get("area"):
            aliases.append(device["area"])

        for alias in aliases:
            if _normalize_value(alias) == normalized_target:
                candidate_devices.append(device)
                break

    if not candidate_devices:
        return None

    if area_hint:
        for device in candidate_devices:
            if device.get("area") and _normalize_value(device["area"]) == _normalize_value(area_hint):
                return device

    return candidate_devices[0]


def _fallback_match_device(transcript: str, controllable_devices: list) -> dict | None:
    normalized_transcript = _normalize_value(transcript)
    area_hint = _extract_area_hint(transcript)

    candidates = []
    for device in controllable_devices:
        aliases = [device.get("name"), device.get("area"), (device.get("entity_id") or "").split(".")[-1].replace("_", " ")]
        for alias in aliases:
            if alias and _normalize_value(alias) in normalized_transcript:
                candidates.append(device)
                break

    if not candidates:
        return None

    if area_hint:
        for device in candidates:
            if device.get("area") and _normalize_value(device["area"]) == _normalize_value(area_hint):
                return device

    return candidates[0]


def _build_intent_templates() -> dict:
    return {
        "language": "en",
        "intents": {
            "HassTurnOn": {
                "data": [
                    {"sentences": [
                        "turn on [the] {name}",
                        "turn on [the] {name} in [the] {area}",
                    ]},
                ]
            },
            "HassTurnOff": {
                "data": [
                    {"sentences": [
                        "turn off [the] {name}",
                        "turn off [the] {name} in [the] {area}",
                    ]},
                ]
            },
        },
    }


def parse_intent_deterministic(transcript: str, controllable_devices: list) -> dict | None:
    """
    Deterministic fallback parser using HassIL with dynamic name/area slots.
    This is intentionally separate from the SLM strategy and follows the same
    public contract as parse_intent().
    """
    if not controllable_devices:
        logger.debug("Deterministic parser received no controllable devices.")
        return None

    intents = Intents.from_dict(_build_intent_templates())
    slot_lists = _build_slot_lists(controllable_devices)
    try:
        result = recognize(transcript, intents, slot_lists=slot_lists)
    except Exception:
        logger.debug("Deterministic parser hit a HassIL match failure; falling back to alias matching for %r", transcript)
        result = None

    if not result:
        logger.debug("Deterministic parser did not match transcript with HassIL, falling back to alias matching: %r", transcript)
        device = _fallback_match_device(transcript, controllable_devices)
        if not device:
            return None
        matched_name = device.get("name") or (device.get("entity_id") or "").split(".")[-1].replace("_", " ")
    else:
        name_entity = result.entities.get("name")
        if not name_entity:
            logger.warning("Deterministic parser matched but produced no device name.")
            return None

        matched_name = name_entity.value
        device = _find_matching_device(transcript, controllable_devices, matched_name)
        if not device:
            logger.warning("Deterministic parser matched %r to no known device.", matched_name)
            return None

    if not device:
        logger.warning("Deterministic parser matched %r to no known device.", matched_name)
        return None

    entity_id = device.get("entity_id")
    if not entity_id or "." not in entity_id:
        logger.warning("Deterministic parser found a device without a valid entity_id: %r", device)
        return None

    entity_domain, _, _ = entity_id.partition(".")
    service = "turn_on" if "turn on" in transcript.lower() or "set " in transcript.lower() else "turn_off"
    if service == "turn_off" and "turn on" in transcript.lower():
        service = "turn_on"

    target_domain = "homeassistant" if entity_domain == "group" else entity_domain
    if entity_domain not in {"switch", "light", "group"}:
        logger.warning("Deterministic parser ignored unsupported entity domain: %s", entity_domain)
        return None

    validated = {
        "domain": target_domain,
        "service": service,
        "entity_id": entity_id,
    }

    brightness_pct = _extract_brightness_pct(transcript)
    if brightness_pct is not None:
        if entity_domain != "light":
            logger.warning("Deterministic parser rejected brightness for non-light entity %s.", entity_id)
            return {}
        if service == "turn_on":
            validated["brightness_pct"] = brightness_pct
        else:
            return {}

    return validated
