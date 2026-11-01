from intents.deterministic_intents import (
    _load_intents,
    parse_intent_deterministic,
)


def test_loads_all_requested_ohf_intent_families():
    intents = _load_intents()
    expected_families = {
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
    }

    assert expected_families <= set(intents.intents)
    assert all(intents.intents[name].data for name in expected_families)


def test_requested_intent_families_delegate_to_home_assistant():
    devices = [
        {"entity_id": "light.bedroom_lamp", "name": "Bedroom Lamp", "area": "Bedroom"},
        {"entity_id": "media_player.living_room", "name": "Living Room Speaker", "area": "Living Room"},
        {"entity_id": "timer.kitchen", "name": "Kitchen Timer", "area": "Kitchen"},
        {"entity_id": "todo.shopping_list", "name": "Shopping List"},
        {"entity_id": "weather.home", "name": "Home"},
    ]
    cases = {
        "pause the living room speaker": "HassMediaPause",
        "turn on the living room speaker": "HassTurnOn",
        "set a timer for five minutes": "HassStartTimer",
        "add apples to my shopping list": "HassListAddItem",
        "what is the weather like": "HassGetWeather",
        "set the bedroom lamp color to red": "HassLightSet",
    }

    for transcript, expected_intent in cases.items():
        assert parse_intent_deterministic(transcript, devices) == {
            "conversation_text": transcript,
            "intent_name": expected_intent,
        }


def test_deterministic_intent_turns_on_named_light():
    devices = [
        {"entity_id": "light.living_room", "name": "Living Room Light", "area": "Living Room"},
        {"entity_id": "switch.plug_1", "name": "Plug 1", "area": "Garage"},
    ]

    intent = parse_intent_deterministic("turn on the living room light", devices)

    assert intent == {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.living_room",
    }


def test_deterministic_intent_uses_ohf_voice_sentence_templates():
    devices = [{"entity_id": "light.living_room", "name": "Living Room Light", "area": "Living Room"}]

    intent = parse_intent_deterministic("activate the living room light", devices)

    assert intent == {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.living_room",
    }


def test_deterministic_intent_adds_custom_sentence_templates(tmp_path, monkeypatch):
    devices = [{"entity_id": "light.living_room", "name": "Living Room Light", "area": "Living Room"}]
    custom_templates = tmp_path / "custom_intents.yaml"
    custom_templates.write_text(
        "language: en\n"
        "intents:\n"
        "  HassTurnOn:\n"
        "    data:\n"
        "      - sentences:\n"
        "          - 'illuminate [the] {name}'\n"
        "        name_domains: [light]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "intents.deterministic_intents._CUSTOM_INTENTS_FILE", custom_templates
    )
    _load_intents.cache_clear()

    try:
        intent = parse_intent_deterministic("illuminate the living room light", devices)
    finally:
        _load_intents.cache_clear()

    assert intent == {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.living_room",
    }


def test_deterministic_intent_supports_group_entities():
    devices = [{"entity_id": "group.living_room_lights", "name": "Living Room Lights", "area": "Living Room"}]

    intent = parse_intent_deterministic("turn on the living room lights", devices)

    assert intent == {
        "domain": "homeassistant",
        "service": "turn_on",
        "entity_id": "group.living_room_lights",
    }


def test_deterministic_intent_extracts_brightness_from_transcript():
    devices = [{"entity_id": "light.kitchen", "name": "Kitchen Light", "area": "Kitchen"}]

    intent = parse_intent_deterministic("set the kitchen light to 42 percent", devices)

    assert intent == {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.kitchen",
        "brightness_pct": 42,
    }


def test_deterministic_intent_ignores_non_light_brightness_requests():
    devices = [{"entity_id": "switch.plug_1", "name": "Plug 1", "area": "Garage"}]

    intent = parse_intent_deterministic("set plug 1 to 50 percent", devices)

    assert intent == {}


def test_deterministic_intent_uses_area_to_find_the_correct_device():
    devices = [
        {"entity_id": "light.kitchen", "name": "Kitchen Light", "area": "Kitchen"},
        {"entity_id": "light.hall", "name": "Kitchen Light", "area": "Hallway"},
    ]

    intent = parse_intent_deterministic("turn on the kitchen light in the hallway", devices)

    assert intent == {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.hall",
    }


def test_deterministic_intent_uses_floor_to_disambiguate_devices():
    devices = [
        {"entity_id": "light.bedroom_downstairs", "name": "Bedroom Lamp", "floor": "Downstairs"},
        {"entity_id": "light.bedroom_upstairs", "name": "Bedroom Lamp", "floor": "Upstairs"},
    ]

    intent = parse_intent_deterministic(
        "turn on the bedroom lamp on the upstairs floor",
        devices,
    )

    assert intent == {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.bedroom_upstairs",
    }
