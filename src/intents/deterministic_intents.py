import copy
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from hassil import Intents
from hassil.intents import RangeSlotList, RangeType, TextSlotList, WildcardSlotList
from hassil.recognize import recognize_all
import yaml

logger = logging.getLogger(__name__)
_SUPPORTED_ENTITY_DOMAINS = {
    "light",
    "switch",
    "group",
    "media_player",
    "timer",
    "todo",
    "weather",
}
# Home Assistant refuses timer intents from clients that aren't voice satellites,
# so timers are run locally instead of via conversation.process.
_LOCAL_TIMER_INTENTS = {
    "HassStartTimer",
    "HassCancelTimer",
    "HassCancelAllTimers",
    "HassIncreaseTimer",
    "HassDecreaseTimer",
    "HassPauseTimer",
    "HassUnpauseTimer",
    "HassTimerStatus",
}
_MEDIA_CONTROL_SERVICES = {
    "HassMediaPause": "media_pause",
    "HassMediaUnpause": "media_play",
    "HassMediaNext": "media_next_track",
    "HassMediaPrevious": "media_previous_track",
}
_CUSTOM_INTENTS_FILE = Path(__file__).with_name("custom_intents.yaml")
_OHF_INTENT_FAMILIES = (
    "HassTurnOn",
    "HassTurnOff",
    "HassLightSet",
    "HassMediaNext",
    "HassMediaPause",
    "HassMediaPlayerMute",
    "HassMediaPlayerUnmute",
    "HassMediaPrevious",
    "HassMediaSearchAndPlay",
    "HassMediaUnpause",
    "HassSetVolume",
    "HassSetVolumeRelative",
    "HassStartTimer",
    "HassCancelTimer",
    "HassCancelAllTimers",
    "HassIncreaseTimer",
    "HassDecreaseTimer",
    "HassPauseTimer",
    "HassUnpauseTimer",
    "HassTimerStatus",
    "HassListAddItem",
    "HassListCompleteItem",
    "HassListRemoveItem",
    "HassShoppingListAddItem",
    "HassShoppingListCompleteItem",
    "HassGetWeather",
)
_DYNAMIC_SLOT_LISTS = {
    "name",
    "ordinal",
    "area",
    "floor",
    "brightness",
    "search_query",
    "shopping_list_item",
    "timer_command",
    "timer_hours",
    "timer_minutes",
    "timer_name",
    "timer_seconds",
    "todo_list_item",
    "volume",
    "volume_step_down",
    "volume_step_up",
    "color_temperature",
}
_ORDINAL_WORDS = (
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
)
_CARDINAL_WORDS = (
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
)


def _ordinal_values() -> list[tuple[str, int]]:
    """Spoken and digit forms ("first", "one", "1st", "1") mapped to their number."""
    values = []
    for number, (ordinal, cardinal) in enumerate(zip(_ORDINAL_WORDS, _CARDINAL_WORDS), 1):
        suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
        values.extend(
            (form, number) for form in (ordinal, cardinal, str(number), f"{number}{suffix}")
        )
    return values


def _find_ohf_intents_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "vendor" / "ohf-voice-intents"
        if (candidate / "sentences" / "en").is_dir():
            return candidate
    return Path(__file__).resolve().parents[2] / "vendor" / "ohf-voice-intents"


_OHF_INTENTS_ROOT = _find_ohf_intents_root()


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


def _build_slot_lists(controllable_devices: list) -> dict[str, Any]:
    name_values = []
    numbered_devices = {}
    area_values = []
    floor_values = []

    for device in controllable_devices:
        entity_id = device.get("entity_id") or ""
        name = device.get("name")
        entity_name = entity_id.split(".")[-1].replace("_", " ") if entity_id else ""
        canonical_name = name or entity_name
        area = device.get("area")
        floor = device.get("floor")

        for alias in dict.fromkeys(value for value in (name, entity_name) if value):
            name_values.append((alias, canonical_name))
            if entity_id.startswith("todo."):
                short_list_alias = re.sub(r"\s+list$", "", alias, flags=re.IGNORECASE).strip()
                if short_list_alias and short_list_alias != alias:
                    name_values.append((short_list_alias, canonical_name))

        numbered_name = re.fullmatch(r"(.+?)\s+(\d+)", canonical_name)
        if numbered_name:
            base_name, number = numbered_name.groups()
            numbered_devices.setdefault(_normalize_value(base_name), []).append(
                (int(number), base_name, canonical_name)
            )

        if area and area not in area_values:
            area_values.append(area)
        if floor and floor not in floor_values:
            floor_values.append(floor)

    for numbered_group in numbered_devices.values():
        for ordinal, (_, base_name, canonical_name) in zip(
            _ORDINAL_WORDS,
            sorted(numbered_group, key=lambda numbered_device: numbered_device[0]),
        ):
            name_values.append((f"{ordinal} {base_name}", canonical_name))

    names = TextSlotList.from_tuples(
        sorted(name_values, key=lambda name_value: len(name_value[0]), reverse=True),
        allow_template=False,
        name="name",
    )
    areas = TextSlotList.from_strings(sorted(area_values, key=len, reverse=True))
    slot_lists: dict[str, Any] = {
        "name": names,
        "area": areas,
        "floor": TextSlotList.from_strings(sorted(floor_values, key=len, reverse=True)),
    }
    slot_lists.update({
        name: TextSlotList.from_tuples(
            sorted(name_values, key=lambda name_value: len(name_value[0]), reverse=True),
            allow_template=False,
            name=name,
        )
        for name in ("timer_name",)
    })
    slot_lists["ordinal"] = TextSlotList.from_tuples(
        _ordinal_values(), allow_template=False, name="ordinal"
    )
    slot_lists.update({
        name: WildcardSlotList(name)
        for name in (
            "search_query", "shopping_list_item", "todo_list_item", "timer_command"
        )
    })
    numeric_slots = {
        "brightness": (1, 100, RangeType.PERCENTAGE),
        "timer_hours": (0, 999, RangeType.NUMBER),
        "timer_minutes": (0, 59, RangeType.NUMBER),
        "timer_seconds": (0, 59, RangeType.NUMBER),
        "volume": (0, 100, RangeType.PERCENTAGE),
        "volume_step_up": (1, 100, RangeType.PERCENTAGE),
        "volume_step_down": (1, 100, RangeType.PERCENTAGE),
        "color_temperature": (1000, 10000, RangeType.TEMPERATURE),
    }
    slot_lists.update({
        name: RangeSlotList(name=name, start=start, stop=stop, type=range_type)
        for name, (start, stop, range_type) in numeric_slots.items()
    })
    return slot_lists


def _extract_area_hint(transcript: str) -> str | None:
    matches = re.findall(r"\bin\s+(?:the\s+)?([a-z0-9][a-z0-9\s-]+?)(?:[.!?]|$)", transcript.lower())
    if not matches:
        return None
    return matches[-1].strip()


def _device_matches_name(device: dict, matched_name: str) -> bool:
    aliases = [device.get("name") or ""]
    if entity_id := device.get("entity_id"):
        aliases.append(entity_id.split(".")[-1].replace("_", " "))
    return any(_normalize_value(alias) == _normalize_value(matched_name) for alias in aliases)


def _find_matching_device(
    transcript: str,
    controllable_devices: list,
    matched_name: str | None,
    matched_floor: str | None = None,
) -> dict | None:
    if not matched_name:
        return None

    area_hint = _extract_area_hint(transcript)

    candidate_devices = [
        device for device in controllable_devices
        if _device_matches_name(device, matched_name)
    ]

    if not candidate_devices:
        return None

    if matched_floor:
        for device in candidate_devices:
            if device.get("floor") and _normalize_value(device["floor"]) == _normalize_value(matched_floor):
                return device

    if area_hint:
        for device in candidate_devices:
            if device.get("area") and _normalize_value(device["area"]) == _normalize_value(area_hint):
                return device

    return candidate_devices[0]


def _merge_mapping(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_mapping(target[key], value)
        elif isinstance(value, list) and isinstance(target.get(key), list):
            target[key].extend(copy.deepcopy(value))
        else:
            target[key] = copy.deepcopy(value)


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as yaml_file:
        value = yaml.safe_load(yaml_file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _supported_ohf_data(data: list, available_slot_lists: set[str]) -> list[dict]:
    supported_data = []
    for item in data:
        item = copy.deepcopy(item)
        required_lists = {
            list_name
            for sentence in item.get("sentences", [])
            for list_name, _slot_name in re.findall(
                r"\{([^}:]+)(?::([^}]+))?\}", sentence
            )
        }
        if missing_lists := required_lists - available_slot_lists:
            logger.debug(
                "Skipping sentence block with unsupported slot lists: %s",
                sorted(missing_lists),
            )
            continue

        domains = item.get("name_domains")
        if domains == "default":
            item["name_domains"] = sorted(_SUPPORTED_ENTITY_DOMAINS)
        elif domains is not None:
            if isinstance(domains, str):
                domains = [domains]
            supported_domains = sorted(set(domains) & _SUPPORTED_ENTITY_DOMAINS)
            if not supported_domains:
                continue
            item["name_domains"] = supported_domains

        supported_data.append(item)
    return supported_data


@lru_cache(maxsize=1)
def _load_intents() -> Intents:
    if not (_OHF_INTENTS_ROOT / "sentences" / "en").is_dir():
        raise RuntimeError(
            "OHF-Voice intents submodule is missing; initialize it with "
            "`git submodule update --init --recursive`."
        )

    intents_data = {
        "language": "en",
        "intents": {},
        "expansion_rules": {},
        "lists": {},
    }

    for folder, target_key in (("rules", "expansion_rules"), ("lists", "lists")):
        for yaml_path in sorted((_OHF_INTENTS_ROOT / folder / "en").glob("*.yaml")):
            file_data = _load_yaml(yaml_path)
            _merge_mapping(intents_data[target_key], file_data.get(target_key, {}))

    sentence_root = _OHF_INTENTS_ROOT / "sentences" / "en"
    available_slot_lists = set(intents_data["lists"]) | _DYNAMIC_SLOT_LISTS
    for intent_name in _OHF_INTENT_FAMILIES:
        intent_data = intents_data["intents"].setdefault(intent_name, {"data": []})
        intent_folder = sentence_root / intent_name
        for yaml_path in sorted(intent_folder.glob("*.yaml")):
            file_data = _load_yaml(yaml_path)
            intent_data["data"].extend(
                _supported_ohf_data(file_data.get("data", []), available_slot_lists)
            )

    common_data = _load_yaml(sentence_root / "_common.yaml")
    intents_data["skip_words"] = common_data.get("skip_words", [])

    if _CUSTOM_INTENTS_FILE.is_file():
        custom_data = _load_yaml(_CUSTOM_INTENTS_FILE)
        if custom_data.get("language", "en") != "en":
            raise ValueError("Custom deterministic intents must use language 'en'.")
        for section in ("expansion_rules", "lists"):
            _merge_mapping(intents_data[section], custom_data.get(section, {}))
        for intent_name, custom_intent in custom_data.get("intents", {}).items():
            target_intent = intents_data["intents"].setdefault(intent_name, {"data": []})
            target_intent["data"].extend(copy.deepcopy(custom_intent.get("data", [])))

    return Intents.from_dict(intents_data)


def parse_intent_deterministic(transcript: str, controllable_devices: list) -> dict | None:
    """
    Deterministic parser using HassIL with dynamic name/area slots.
    """
    intents = _load_intents()
    slot_lists = _build_slot_lists(controllable_devices)
    try:
        result = None
        first_result = None
        first_typed_media_result = None
        for candidate in recognize_all(transcript, intents, slot_lists=slot_lists):
            if first_result is None:
                first_result = candidate
            if (
                first_typed_media_result is None
                and candidate.intent.name == "HassMediaSearchAndPlay"
                and candidate.entities.get("media_class")
            ):
                first_typed_media_result = candidate
            name_entity = candidate.entities.get("name")
            if name_entity and _find_matching_device(
                transcript,
                controllable_devices,
                name_entity.value,
                candidate.entities.get("floor").value
                if candidate.entities.get("floor") else None,
            ):
                result = candidate
                break
        if result is None:
            result = first_typed_media_result or first_result
    except Exception:
        logger.debug("Deterministic parser hit a HassIL match failure; falling back to alias matching.")
        result = None

    if not result:
        logger.debug("Deterministic parser did not match transcript.")
        return None

    intent_name = result.intent.name
    if intent_name in _LOCAL_TIMER_INTENTS:
        seconds = sum(
            float(result.entities[slot].value) * factor
            for slot, factor in (("hours", 3600), ("minutes", 60), ("seconds", 1))
            if slot in result.entities
        )
        timer = {"timer_action": intent_name, "timer_seconds": seconds}
        if number_entity := result.entities.get("timer_number"):
            timer["timer_number"] = int(number_entity.value)
        start_seconds = sum(
            float(result.entities[slot].value) * factor
            for slot, factor in (("start_hours", 3600), ("start_minutes", 60), ("start_seconds", 1))
            if slot in result.entities
        )
        if start_seconds:
            timer["timer_start_seconds"] = start_seconds
        if name_entity := result.entities.get("name"):
            timer["timer_name"] = str(name_entity.value)
        return timer

    if intent_name == "HassGetWeather":
        name_entity = result.entities.get("name")
        weather_devices = [
            device for device in controllable_devices
            if (device.get("entity_id") or "").startswith("weather.")
        ]
        weather_device = (
            _find_matching_device(transcript, weather_devices, name_entity.value)
            if name_entity else (weather_devices[0] if weather_devices else None)
        )
        if weather_device:
            weather_intent = {
                "weather_entity_id": weather_device["entity_id"],
                "weather_name": weather_device.get("name", weather_device["entity_id"]),
            }
            if temperature_unit := weather_device.get("temperature_unit"):
                weather_intent["temperature_unit"] = temperature_unit
            return weather_intent

        return {"conversation_text": transcript, "intent_name": intent_name}

    if intent_name == "PersonalAssistantGetTodoItems":
        name_entity = result.entities.get("name")
        if not name_entity:
            return None
        device = _find_matching_device(
            transcript, controllable_devices, name_entity.value
        )
        if not device or not (device_entity_id := device.get("entity_id", "")).startswith("todo."):
            return None
        return {
            "todo_items_entity_id": device_entity_id,
            "todo_list_name": device.get("name", device_entity_id),
        }

    if intent_name == "HassMediaSearchAndPlay":
        intent = {"conversation_text": transcript, "intent_name": intent_name}
        media_class = result.entities.get("media_class")
        if media_class:
            media_type = str(media_class.value)
            if media_type not in {"artist", "album", "track", "playlist"}:
                return intent
            intent["media_search_type"] = media_type
        if search_entity := result.entities.get("search_query"):
            intent["media_search_query"] = str(search_entity.value).strip()
        if name_entity := result.entities.get("name"):
            intent["media_player_name"] = str(name_entity.value)
        return intent

    if intent_name in _MEDIA_CONTROL_SERVICES and (name_entity := result.entities.get("name")):
        area_entity = result.entities.get("area")
        floor_entity = result.entities.get("floor")
        area = str(area_entity.value) if area_entity else _extract_area_hint(transcript)
        floor = str(floor_entity.value) if floor_entity else None
        players = [
            device for device in controllable_devices
            if (device.get("entity_id") or "").startswith("media_player.")
            and device.get("platform") == "music_assistant"
            and _device_matches_name(device, str(name_entity.value))
            and (not area or _normalize_value(device.get("area") or "") == _normalize_value(area))
            and (not floor or _normalize_value(device.get("floor") or "") == _normalize_value(floor))
        ]
        if len(players) == 1:
            return {
                "media_control_entity_id": players[0]["entity_id"],
                "media_control_service": _MEDIA_CONTROL_SERVICES[intent_name],
            }

    if intent_name not in {"HassTurnOn", "HassTurnOff", "HassLightSet"}:
        return {"conversation_text": transcript, "intent_name": intent_name}

    if intent_name == "HassLightSet" and not result.entities.get("brightness"):
        return {"conversation_text": transcript, "intent_name": intent_name}

    name_entity = result.entities.get("name")
    if not name_entity:
        return {"conversation_text": transcript, "intent_name": intent_name}

    floor_entity = result.entities.get("floor")
    device = _find_matching_device(
        transcript,
        controllable_devices,
        name_entity.value,
        floor_entity.value if floor_entity else None,
    )
    if not device:
        logger.warning("Deterministic parser matched %r to no known device.", name_entity.value)
        return None

    entity_id = device.get("entity_id")
    if not entity_id or "." not in entity_id:
        logger.warning("Deterministic parser found a device without a valid entity_id: %r", device)
        return None

    entity_domain, _, _ = entity_id.partition(".")
    if intent_name == "HassTurnOff":
        service = "turn_off"
    elif intent_name in {"HassTurnOn", "HassLightSet"}:
        service = "turn_on"
    else:
        logger.warning("Deterministic parser matched unsupported intent: %s", intent_name)
        return None

    target_domain = "homeassistant" if entity_domain == "group" else entity_domain
    if entity_domain not in {"switch", "light", "group"}:
        if entity_domain in _SUPPORTED_ENTITY_DOMAINS:
            return {"conversation_text": transcript, "intent_name": intent_name}
        logger.warning("Deterministic parser ignored unsupported entity domain: %s", entity_domain)
        return None

    validated = {
        "domain": target_domain,
        "service": service,
        "entity_id": entity_id,
    }

    brightness_entity = result.entities.get("brightness")
    try:
        brightness_pct = int(brightness_entity.value) if brightness_entity else None
    except (TypeError, ValueError):
        brightness_pct = None
    if brightness_pct is None:
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
