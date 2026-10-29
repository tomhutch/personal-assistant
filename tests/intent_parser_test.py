import pytest
import requests
from unittest.mock import patch, MagicMock
from intent_parser import parse_intent

class TestParseIntent:
    @patch("requests.post", side_effect=requests.ConnectionError("connection refused"))
    def test_slm_connection_failure_returns_none(self, mock_post):
        assert parse_intent("turn on the light", []) is None

    @patch("requests.post")
    def test_scenario_valid_intent_parsing(self, mock_post):
        """Scenario: Qwen SLM correctly maps transcript and device list to a valid action JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"domain": "switch", "service": "turn_off", "entity_id": "switch.sonoff_s60zbtpg"}'
                }
            }]
        }
        mock_post.return_value = mock_response
        
        devices = [{"entity_id": "switch.sonoff_s60zbtpg", "name": "Plug 1"}]
        intent = parse_intent("turn off first plug", devices)
        
        assert intent["domain"] == "switch"
        assert intent["service"] == "turn_off"
        assert intent["entity_id"] == "switch.sonoff_s60zbtpg"

    @patch("requests.post")
    def test_light_brightness_intent_is_validated(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": (
                        '{"domain":"light","service":"turn_on",'
                        '"entity_id":"light.living_room","brightness_pct":45}'
                    )
                }
            }]
        }
        mock_post.return_value = mock_response
        devices = [{
            "entity_id": "light.living_room",
            "name": "Ceiling Light",
            "area": "Living Room",
        }]

        intent = parse_intent("set the living room light to 45 percent", devices)

        assert intent == {
            "domain": "light",
            "service": "turn_on",
            "entity_id": "light.living_room",
            "brightness_pct": 45,
        }
        assert '"area": "Living Room"' in mock_post.call_args.kwargs["json"]["messages"][1]["content"]

    @patch("requests.post")
    def test_explicit_transcript_percentage_fills_missing_brightness(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": (
                        '{"domain":"light","service":"turn_on",'
                        '"entity_id":"light.living_room"}'
                    )
                }
            }]
        }
        mock_post.return_value = mock_response

        intent = parse_intent(
            "set the living room light to 35 percent",
            [{"entity_id": "light.living_room", "name": "Living Room Light"}],
        )

        assert intent["brightness_pct"] == 35

    @patch("requests.post")
    def test_group_intent_uses_homeassistant_service_domain(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": (
                        '{"domain":"homeassistant","service":"turn_on",'
                        '"entity_id":"group.living_room_lights"}'
                    )
                }
            }]
        }
        mock_post.return_value = mock_response
        devices = [{
            "entity_id": "group.living_room_lights",
            "name": "Living Room Lights",
            "area": "Living Room",
            "members": ["light.ceiling", "light.lamp"],
        }]

        intent = parse_intent("turn on the living room lights", devices)

        assert intent == {
            "domain": "homeassistant",
            "service": "turn_on",
            "entity_id": "group.living_room_lights",
        }

    @patch("requests.post")
    def test_light_entity_domain_overrides_slm_domain(self, mock_post, caplog):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": (
                        '{"domain":"homeassistant","service":"turn_on",'
                        '"entity_id":"light.living_room_lights"}'
                    )
                }
            }]
        }
        mock_post.return_value = mock_response

        intent = parse_intent(
            "turn on the living room lights",
            [{"entity_id": "light.living_room_lights", "name": "Living Room Lights"}],
        )

        assert intent == {
            "domain": "light",
            "service": "turn_on",
            "entity_id": "light.living_room_lights",
        }
        assert "using inferred service domain light" in caplog.text

    @patch("requests.post")
    def test_rejects_brightness_for_switch(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": (
                        '{"domain":"switch","service":"turn_on",'
                        '"entity_id":"switch.plug_1","brightness_pct":45}'
                    )
                }
            }]
        }
        mock_post.return_value = mock_response

        intent = parse_intent(
            "set the plug to 45 percent",
            [{"entity_id": "switch.plug_1", "name": "Plug 1"}],
        )

        assert intent == {}

    @patch("requests.post")
    def test_scenario_malformed_slm_response(self, mock_post):
        """Scenario: Qwen SLM returns invalid markdown/JSON; parser handles it safely without crashing."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Sorry, I didn't understand that command."
                }
            }]
        }
        mock_post.return_value = mock_response
        
        devices = [{"entity_id": "switch.sonoff_s60zbtpg", "name": "Plug 1"}]
        intent = parse_intent("do a backflip", devices)
        
        # Should return None or a safe fallback dict instead of raising a JSON decode error
        assert intent is None or isinstance(intent, dict)