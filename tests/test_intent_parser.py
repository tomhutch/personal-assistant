from unittest.mock import patch

import pytest

from intent_parser import parse_intent


@pytest.mark.parametrize(
    ("transcript", "service"),
    [
        ("turn on the Living Room Lights.", "turn_on"),
        ("turn off the living room lights.", "turn_off"),
        ("Hey Jarvis, turn on the Living Room Lights.", "turn_on"),
        ("Hey, turn off the living room lights.", "turn_off"),
        ("  HEY JARVIS turn on the Living Room Lights.", "turn_on"),
    ],
)
def test_logged_living_room_light_commands(transcript, service):
    devices = [
        {
            "entity_id": "light.living_room_lights",
            "name": "Living Room Lights",
            "area": "Living Room",
        }
    ]

    with patch("config.USE_DETERMINISTIC_INTENT", True):
        intent = parse_intent(transcript, devices)

    assert intent == {
        "domain": "light",
        "service": service,
        "entity_id": "light.living_room_lights",
    }


def test_prefixed_conversation_intent_passes_clean_command_to_home_assistant():
    with patch("config.USE_DETERMINISTIC_INTENT", True):
        intent = parse_intent("Hey Jarvis, what's the weather?", [])

    assert intent == {
        "conversation_text": "what's the weather?",
        "intent_name": "HassGetWeather",
    }


def test_ai_parser_receives_original_command_after_deterministic_match():
    devices = [{"entity_id": "light.living_room_lights", "name": "Living Room Lights"}]
    expected = {"domain": "light", "service": "turn_on", "entity_id": "light.living_room_lights"}
    with (
        patch("config.USE_DETERMINISTIC_INTENT", False),
        patch("intent_parser.parse_intent_ai", return_value=expected) as parse_ai,
    ):
        intent = parse_intent("Hey Jarvis, turn on the Living Room Lights.", devices)

    assert intent == expected
    parse_ai.assert_called_once_with("Hey Jarvis, turn on the Living Room Lights.", devices)


def test_matching_original_transcript_is_not_stripped():
    expected = {"conversation_text": "Hey, play Hey Jude", "media_search_query": "Hey Jude"}
    with (
        patch("config.USE_DETERMINISTIC_INTENT", True),
        patch("intent_parser.parse_intent_deterministic", return_value=expected) as parse_deterministic,
    ):
        assert parse_intent("Hey, play Hey Jude", []) == expected
    parse_deterministic.assert_called_once_with("Hey, play Hey Jude", [])


@pytest.mark.parametrize("transcript", ["Hey Jarvis,", "Hey,"])
def test_prefix_without_a_command_does_not_match(transcript):
    with patch("config.USE_DETERMINISTIC_INTENT", True):
        assert parse_intent(transcript, []) is None


def test_prefixed_music_search_preserves_song_title():
    devices = [{"entity_id": "media_player.sonos", "name": "Sonos"}]
    with patch("config.USE_DETERMINISTIC_INTENT", True):
        intent = parse_intent("Hey Jarvis, play Hey Jude on Sonos", devices)

    assert intent is not None
    assert intent["media_search_query"] == "Hey Jude"
    assert intent["conversation_text"] == "play Hey Jude on Sonos"


@pytest.mark.parametrize(
    "transcript", ["play Hey Jarvis on Sonos", "Hey Jarvis, play Hey Jarvis on Sonos"]
)
def test_music_search_preserves_wake_phrase_in_title(transcript):
    devices = [{"entity_id": "media_player.sonos", "name": "Sonos"}]
    with patch("config.USE_DETERMINISTIC_INTENT", True):
        intent = parse_intent(transcript, devices)

    assert intent is not None
    assert intent["media_search_query"] == "Hey Jarvis"
    assert intent["conversation_text"] == "play Hey Jarvis on Sonos"


@pytest.mark.parametrize("transcript", ["turn on Hey Jarvis", "Hey Jarvis, turn on Hey Jarvis"])
def test_light_name_can_contain_wake_phrase(transcript):
    devices = [{"entity_id": "light.hey_jarvis", "name": "Hey Jarvis"}]
    with patch("config.USE_DETERMINISTIC_INTENT", True):
        intent = parse_intent(transcript, devices)

    assert intent == {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.hey_jarvis",
    }


def test_configured_wake_word_is_used_for_prefix():
    devices = [{"entity_id": "light.lamp", "name": "Lamp"}]
    with (
        patch("config.USE_DETERMINISTIC_INTENT", True),
        patch("config.WAKE_WORD_MODEL", "hey_mycroft"),
    ):
        intent = parse_intent("Hey Mycroft, turn on the lamp", devices)

    assert intent == {"domain": "light", "service": "turn_on", "entity_id": "light.lamp"}


def test_failed_prefixed_command_is_parsed_once():
    with (
        patch("config.USE_DETERMINISTIC_INTENT", True),
        patch("intent_parser.parse_intent_deterministic", return_value=None) as parse_deterministic,
    ):
        assert parse_intent("Hey Jarvis, something unknown", []) is None
    parse_deterministic.assert_called_once_with("Hey Jarvis, something unknown", [])


def test_unmatched_prefixed_text_is_not_stripped_before_ai_parser():
    with (
        patch("config.USE_DETERMINISTIC_INTENT", False),
        patch("intent_parser.parse_intent_ai", return_value=None) as parse_ai,
    ):
        assert parse_intent("Hey, something unknown", []) is None
    parse_ai.assert_called_once_with("Hey, something unknown", [])


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


def test_volume_control_uses_deterministic_parser_with_ai_enabled():
    intent = {
        "media_control_entity_id": "media_player.kitchen_sonos",
        "media_control_service": "volume_set",
        "media_control_data": {"volume_level": 0.3},
    }
    with (
        patch("config.USE_DETERMINISTIC_INTENT", False),
        patch("intent_parser.parse_intent_deterministic", return_value=intent),
        patch("intent_parser.parse_intent_ai") as parse_ai,
    ):
        assert parse_intent("set Sonos volume to 30 percent", []) == intent
    parse_ai.assert_not_called()