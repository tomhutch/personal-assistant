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

    @patch("requests.post")
    def test_scenario_execute_command_success(self, mock_post):
        """Scenario: Successfully fires a service call to Home Assistant to turn off a device."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        success = execute_ha_command(domain="switch", service="turn_off", entity_id="switch.sonoff_s60zbtpg")
        
        assert success is True
        mock_post.assert_called_once()