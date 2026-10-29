import pytest
from unittest.mock import patch, MagicMock
from ha_client import get_ha_devices, execute_ha_command

class TestHomeAssistantClient:
    @patch("requests.get")
    def test_scenario_fetch_controllable_devices(self, mock_get):
        """Scenario: Successfully queries HA states API and filters for controllable switches/lights."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"entity_id": "switch.sonoff_s60zbtpg", "attributes": {"friendly_name": "Plug 1"}, "state": "on"},
            {"entity_id": "sensor.temperature", "attributes": {"friendly_name": "Temp Sensor"}, "state": "21"},
            {"entity_id": "light.living_room", "attributes": {"friendly_name": "Living Room Light"}, "state": "off"}
        ]
        mock_get.return_value = mock_response
        
        devices = get_ha_devices()
        
        # Should filter out non-controllable sensors and keep switches/lights
        assert len(devices) == 2
        assert devices[0]["entity_id"] == "switch.sonoff_s60zbtpg"

    @patch("requests.get")
    def test_fetch_unauthorized_logs_and_returns_none(self, mock_get, caplog):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        assert get_ha_devices() is None
        assert "Failed to fetch HA states. Status: 401" in caplog.text

    @patch("requests.get")
    def test_fetch_excludes_network_led_entities(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "entity_id": "switch.plug_2_network_led",
                "attributes": {"friendly_name": "Plug 2 Network LED"},
                "state": "on",
            },
            {
                "entity_id": "switch.plug_2",
                "attributes": {"friendly_name": "Plug 2"},
                "state": "off",
            },
            {
                "entity_id": "light.network_status",
                "attributes": {"friendly_name": "Network_LED"},
                "state": "on",
            },
        ]
        mock_get.return_value = mock_response

        devices = get_ha_devices()

        assert [device["entity_id"] for device in devices] == ["switch.plug_2"]

    @patch("requests.post")
    def test_scenario_execute_command_success(self, mock_post):
        """Scenario: Successfully fires a service call to Home Assistant to turn off a device."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'[{"entity_id":"switch.sonoff_s60zbtpg","state":"off"}]'
        mock_response.json.return_value = [
            {"entity_id": "switch.sonoff_s60zbtpg", "state": "off"}
        ]
        mock_post.return_value = mock_response
        
        success = execute_ha_command(domain="switch", service="turn_off", entity_id="switch.sonoff_s60zbtpg")
        
        assert success is True
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_empty_success_response_is_command_failure(self, mock_post, caplog):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"[]"
        mock_response.json.return_value = []
        mock_post.return_value = mock_response

        success = execute_ha_command(
            domain="switch", service="turn_on", entity_id="switch.plug_2"
        )

        assert success is False
        assert "returned HTTP 200 but reported no state change" in caplog.text

    @patch("requests.post")
    def test_light_brightness_is_sent_to_home_assistant(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'[{"entity_id":"light.living_room","state":"on"}]'
        mock_response.json.return_value = [
            {"entity_id": "light.living_room", "state": "on"}
        ]
        mock_post.return_value = mock_response

        success = execute_ha_command(
            "light", "turn_on", "light.living_room", brightness_pct=45
        )

        assert success is True
        assert mock_post.call_args.kwargs["json"] == {
            "entity_id": "light.living_room",
            "brightness_pct": 45,
        }