from unittest.mock import patch

from intent_parser import parse_intent


def test_entry_point_uses_deterministic_parser_when_enabled():
    devices = [
        {
            "entity_id": "light.living_room",
            "name": "Living Room Light",
            "area": "Living Room",
        }
    ]

    with patch("config.USE_DETERMINISTIC_INTENT", True):
        intent = parse_intent("turn on the living room light", devices)

    assert intent == {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.living_room",
    }