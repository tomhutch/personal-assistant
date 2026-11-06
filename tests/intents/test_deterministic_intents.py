import pytest

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
        {"entity_id": "weather.forecast_home", "name": "Home"},
    ]
    cases = {
        "pause the living room speaker": "HassMediaPause",
        "turn on the living room speaker": "HassTurnOn",
        "add apples to my shopping list": "HassListAddItem",
        "set the bedroom lamp color to red": "HassLightSet",
    }

    for transcript, expected_intent in cases.items():
        assert parse_intent_deterministic(transcript, devices) == {
            "conversation_text": transcript,
            "intent_name": expected_intent,
        }


def test_timer_intents_are_handled_locally():
    assert parse_intent_deterministic("set a timer for five minutes", []) == {
        "timer_action": "HassStartTimer",
        "timer_seconds": 300.0,
    }
    assert parse_intent_deterministic("cancel all timers", [])["timer_action"] == "HassCancelAllTimers"


def test_timers_can_be_targeted_by_number_or_duration():
    for transcript in ("cancel the first timer", "cancel timer one", "cancel timer number 1"):
        assert parse_intent_deterministic(transcript, [])["timer_number"] == 1
    assert parse_intent_deterministic("pause the second timer", [])["timer_number"] == 2
    assert parse_intent_deterministic("cancel the 10 minute timer", [])["timer_start_seconds"] == 600.0


def test_generic_weather_query_uses_a_discovered_weather_entity():
    devices = [
        {"entity_id": "weather.forecast_home", "name": "Home"},
        {"entity_id": "light.kitchen", "name": "Kitchen Light"},
    ]

    assert parse_intent_deterministic("What's the weather?", devices) == {
        "weather_entity_id": "weather.forecast_home",
        "weather_name": "Home",
    }


def test_weather_intent_matches_without_discovered_devices():
    transcript = "What's the weather?"

    assert parse_intent_deterministic(transcript, []) == {
        "conversation_text": transcript,
        "intent_name": "HassGetWeather",
    }


def test_custom_todo_query_resolves_named_shopping_list():
    devices = [{"entity_id": "todo.shopping_list", "name": "Shopping List"}]

    intent = parse_intent_deterministic("what's on the shopping list", devices)

    assert intent == {
        "todo_items_entity_id": "todo.shopping_list",
        "todo_list_name": "Shopping List",
    }


def test_custom_todo_query_requires_the_word_list():
    devices = [{"entity_id": "todo.shopping_list", "name": "Shopping List"}]

    assert parse_intent_deterministic("what's on the shopping", devices) is None


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


def test_deterministic_intent_resolves_first_second_and_third_plug():
    devices = [
        {"entity_id": "switch.plug_1", "name": "Plug 1"},
        {"entity_id": "switch.plug_2", "name": "Plug 2"},
        {"entity_id": "switch.plug_3", "name": "Plug 3"},
    ]

    for ordinal, entity_id in (
        ("first", "switch.plug_1"),
        ("second", "switch.plug_2"),
        ("third", "switch.plug_3"),
    ):
        assert parse_intent_deterministic(f"turn off {ordinal} plug", devices) == {
            "domain": "switch",
            "service": "turn_off",
            "entity_id": entity_id,
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


def test_media_search_intent_exposes_query_and_player_name():
    devices = [{"entity_id": "media_player.kitchen_kitchen", "name": "Sonos"}]

    intent = parse_intent_deterministic("play Thunderstruck on Sonos", devices)

    assert intent["intent_name"] == "HassMediaSearchAndPlay"
    assert intent["media_search_query"] == "Thunderstruck"
    assert intent["media_player_name"] == "Sonos"
    assert intent["conversation_text"] == "play Thunderstruck on Sonos"


@pytest.mark.parametrize(
    ("phrase", "query", "media_type"),
    [
        ("play the album with you here on Sonos", "with you here", "album"),
        ("play the song thunderstruck on Sonos", "thunderstruck", "track"),
        ("play the track thunderstruck on Sonos", "thunderstruck", "track"),
        ("play the artist ACDC on Sonos", "ACDC", "artist"),
        ("play the playlist rock on Sonos", "rock", "playlist"),
        ("play the album back in black", "back in black", "album"),
    ],
)
def test_media_search_preserves_explicit_media_class(phrase, query, media_type):
    devices = [{"entity_id": "media_player.kitchen_sonos", "name": "Sonos"}]
    intent = parse_intent_deterministic(phrase, devices)
    assert intent["media_search_query"] == query
    assert intent["media_search_type"] == media_type


def test_non_music_media_class_delegates_to_home_assistant():
    phrase = "play the movie heat on Sonos"
    devices = [{"entity_id": "media_player.kitchen_sonos", "name": "Sonos"}]
    assert parse_intent_deterministic(phrase, devices) == {
        "conversation_text": phrase,
        "intent_name": "HassMediaSearchAndPlay",
    }
