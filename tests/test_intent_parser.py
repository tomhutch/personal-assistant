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


def test_music_search_uses_deterministic_slots_with_ai_parser_enabled():
    media_intent = {
        "conversation_text": "play Led Zeppelin on Sonos",
        "media_search_query": "Led Zeppelin",
        "media_player_name": "Sonos",
    }

    with (
        patch("config.USE_DETERMINISTIC_INTENT", False),
        patch("intent_parser.parse_intent_deterministic", return_value=media_intent),
        patch("intent_parser.parse_intent_ai") as parse_ai,
    ):
        intent = parse_intent("play Led Zeppelin on Sonos", [])

    assert intent == media_intent
    parse_ai.assert_not_called()


def test_named_music_control_uses_deterministic_parser_with_ai_enabled():
    media_intent = {
        "media_control_entity_id": "media_player.kitchen_sonos",
        "media_control_service": "media_pause",
    }
    with (
        patch("config.USE_DETERMINISTIC_INTENT", False),
        patch("intent_parser.parse_intent_deterministic", return_value=media_intent),
        patch("intent_parser.parse_intent_ai") as parse_ai,
    ):
        assert parse_intent("pause Sonos", []) == media_intent
    parse_ai.assert_not_called()


def test_ambiguous_music_control_delegates_to_conversation_with_ai_enabled():
    intent = {"conversation_text": "pause Sonos", "intent_name": "HassMediaPause"}
    with (
        patch("config.USE_DETERMINISTIC_INTENT", False),
        patch("intent_parser.parse_intent_deterministic", return_value=intent),
        patch("intent_parser.parse_intent_ai") as parse_ai,
    ):
        assert parse_intent("pause Sonos", []) == intent
    parse_ai.assert_not_called()