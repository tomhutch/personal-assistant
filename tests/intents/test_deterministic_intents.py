from intents.deterministic_intents import parse_intent_deterministic


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
