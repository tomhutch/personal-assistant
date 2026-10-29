import pytest
import json
from unittest.mock import patch, MagicMock
from ha_client import _fetch_entity_areas, get_ha_devices, execute_ha_command

class TestHomeAssistantClient:
    @patch("ha_client.websocket.create_connection")
    def test_fetch_entity_areas_joins_entity_and_area_registries(
        self, mock_create_connection, monkeypatch
    ):
        monkeypatch.setattr("ha_client.HA_URL", "http://ha.example:8123")
        monkeypatch.setattr("ha_client.HA_TOKEN", "test-token")
        monkeypatch.setattr("ha_client._area_cache", {})
        monkeypatch.setattr("ha_client._area_cache_expires_at", 0)
        connection = MagicMock()
        connection.recv.side_effect = [
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            json.dumps({
                "id": 1,
                "type": "result",
                "success": True,
                "result": [{"entity_id": "light.living_room", "area_id": "living"}],
            }),
            json.dumps({
                "id": 2,
                "type": "result",
                "success": True,
                "result": [{"area_id": "living", "name": "Living Room"}],
            }),
        ]
        mock_create_connection.return_value = connection

        assert _fetch_entity_areas() == {"light.living_room": "Living Room"}
        mock_create_connection.assert_called_once_with(
            "ws://ha.example:8123/api/websocket", timeout=2.0
        )
        connection.close.assert_called_once()

    @patch("ha_client._fetch_entity_areas", return_value={"light.living_room": "Living Room"})
    @patch("requests.get")
    def test_scenario_fetch_controllable_devices(self, mock_get, mock_areas):
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
        assert devices[1]["area"] == "Living Room"

    @patch("requests.get")
    def test_fetch_unauthorized_logs_and_returns_none(self, mock_get, caplog):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        assert get_ha_devices() is None
        assert "Failed to fetch HA states. Status: 401" in caplog.text

    @patch("ha_client._fetch_entity_areas", return_value={})
    @patch("requests.get")
    def test_fetch_excludes_network_led_entities(self, mock_get, mock_areas):
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

    @patch("ha_client._fetch_entity_areas", return_value={"group.living_room_lights": "Living Room"})
    @patch("requests.get")
    def test_fetch_includes_controllable_groups(self, mock_get, mock_areas):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{
            "entity_id": "group.living_room_lights",
            "attributes": {
                "friendly_name": "Living Room Lights",
                "entity_id": ["light.ceiling", "switch.lamp_plug"],
            },
            "state": "off",
        }]
        mock_get.return_value = mock_response

        devices = get_ha_devices()

        assert devices == [{
            "entity_id": "group.living_room_lights",
            "name": "Living Room Lights",
            "state": "off",
            "members": ["light.ceiling", "switch.lamp_plug"],
            "area": "Living Room",
        }]

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

    @patch("requests.post")
    def test_group_command_accepts_member_state_changes(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'[{"entity_id":"light.ceiling","state":"on"}]'
        mock_response.json.return_value = [
            {"entity_id": "light.ceiling", "state": "on"}
        ]
        mock_post.return_value = mock_response

        assert execute_ha_command(
            "homeassistant", "turn_on", "group.living_room_lights"
        ) is True
        mock_post.assert_called_once()