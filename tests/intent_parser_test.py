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