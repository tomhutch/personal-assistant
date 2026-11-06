import pytest
import json
import requests
import ha_client
from unittest.mock import patch, MagicMock
from ha_client import (
    _fetch_entity_areas,
    execute_ha_command,
    get_ha_todo_items,
    get_ha_devices,
    get_ha_weather_forecast,
    process_ha_conversation,
)

class TestHomeAssistantClient:
    @pytest.mark.parametrize(("service", "data"), [
        ("volume_set", {"volume_level": 0.3}),
        ("volume_set", {"volume_level": 0.0}),
        ("volume_set", {"volume_level": 1.0}),
        ("volume_mute", {"is_volume_muted": True}),
        ("volume_mute", {"is_volume_muted": False}),
        ("volume_up", {}),
        ("volume_down", {}),
    ])
    @patch("ha_client.requests.post")
    def test_volume_control_sends_expected_payload(self, mock_post, service, data):
        mock_post.return_value.status_code = 200
        assert ha_client.control_music_assistant_player("media_player.sonos", service, data) is True
        assert mock_post.call_args.args[0].endswith(f"/api/services/media_player/{service}")
        assert mock_post.call_args.kwargs["json"] == {"entity_id": "media_player.sonos", **data}

    @pytest.mark.parametrize(("service", "data"), [
        ("volume_set", {"volume_level": -0.1}),
        ("volume_set", {"volume_level": 1.1}),
        ("volume_set", {"volume_level": True}),
        ("volume_set", {"volume_level": float("nan")}),
        ("volume_mute", {"is_volume_muted": 1}),
        ("volume_up", {"entity_id": "media_player.other"}),
    ])
    @patch("ha_client.requests.post")
    def test_volume_control_rejects_invalid_data(self, mock_post, service, data):
        assert ha_client.control_music_assistant_player("media_player.sonos", service, data) is False
        mock_post.assert_not_called()

    @pytest.mark.parametrize("service", ["media_pause", "media_play", "media_next_track", "media_previous_track"])
    @patch("ha_client.requests.post")
    def test_control_music_assistant_player_calls_ha_service(self, mock_post, service, monkeypatch):
        monkeypatch.setattr("ha_client.HA_TOKEN", "test-token")
        mock_post.return_value.status_code = 200
        assert ha_client.control_music_assistant_player("media_player.kitchen_sonos", service) is True
        args, kwargs = mock_post.call_args
        assert args[0].endswith(f"/api/services/media_player/{service}")
        assert kwargs["json"] == {"entity_id": "media_player.kitchen_sonos"}
        assert kwargs["headers"]["Authorization"] == "Bearer test-token"

    @patch("ha_client.requests.post")
    def test_control_music_assistant_player_handles_failure_and_timeout(self, mock_post):
        mock_post.return_value.status_code = 500
        assert ha_client.control_music_assistant_player("media_player.sonos", "media_pause") is False
        mock_post.side_effect = requests.ReadTimeout("timeout")
        assert ha_client.control_music_assistant_player("media_player.sonos", "media_pause") is None

    @patch("ha_client.websocket.create_connection")
    def test_fetch_entity_areas_joins_entity_and_area_registries(
        self, mock_create_connection, monkeypatch
    ):
        monkeypatch.setattr("ha_client.HA_URL", "http://ha.example:8123")
        monkeypatch.setattr("ha_client.HA_TOKEN", "test-token")
        monkeypatch.setattr("ha_client._area_cache", {})
        monkeypatch.setattr("ha_client._entity_floors", {})
        monkeypatch.setattr("ha_client._area_cache_expires_at", 0)
        connection = MagicMock()
        connection.recv.side_effect = [
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            json.dumps({
                "id": 1,
                "type": "result",
                "success": True,
                "result": [{
                    "entity_id": "light.living_room",
                    "area_id": "living",
                    "platform": "music_assistant",
                    "config_entry_id": "entry-id",
                }],
            }),
            json.dumps({
                "id": 2,
                "type": "result",
                "success": True,
                "result": [{"area_id": "living", "name": "Living Room", "floor_id": "downstairs"}],
            }),
            json.dumps({
                "id": 3,
                "type": "result",
                "success": True,
                "result": [{"floor_id": "downstairs", "name": "Downstairs"}],
            }),
        ]
        mock_create_connection.return_value = connection

        assert _fetch_entity_areas() == {"light.living_room": "Living Room"}
        assert ha_client._entity_floors == {"light.living_room": "Downstairs"}
        assert ha_client._entity_platforms == {
            "light.living_room": "music_assistant"
        }
        assert ha_client._entity_config_entry_ids == {
            "light.living_room": "entry-id"
        }
        mock_create_connection.assert_called_once_with(
            "ws://ha.example:8123/api/websocket", timeout=2.0
        )
        connection.close.assert_called_once()

    @pytest.mark.parametrize(
        ("media_type", "expected_types"),
        [
            ("album", ["album"]),
            ("track", ["track"]),
            ("playlist", ["playlist"]),
            ("artist", ["artist", "playlist"]),
        ],
    )
    @patch("ha_client._websocket_result", return_value={"response": {}})
    @patch("ha_client._create_authenticated_websocket")
    def test_search_music_assistant_uses_requested_type(
        self, mock_connection, mock_result, media_type, expected_types, monkeypatch
    ):
        monkeypatch.setattr("ha_client.HA_TOKEN", "test-token")
        connection = MagicMock()
        mock_connection.return_value = connection

        ha_client.search_music_assistant("test query", "entry-id", media_type)

        request = json.loads(connection.send.call_args.args[0])
        assert request["service_data"]["media_type"] == expected_types

    @patch("ha_client._websocket_result", return_value={"response": {"artists": []}})
    @patch("ha_client._create_authenticated_websocket")
    def test_search_music_assistant_calls_search_service(
        self, mock_connection, mock_result, monkeypatch
    ):
        monkeypatch.setattr("ha_client.HA_TOKEN", "test-token")
        connection = MagicMock()
        mock_connection.return_value = connection

        assert ha_client.search_music_assistant("Led Zeppelin", "entry-id") == {
            "artists": []
        }

        request = json.loads(connection.send.call_args.args[0])
        assert request["domain"] == "music_assistant"
        assert request["service"] == "search"
        assert request["service_data"] == {
            "config_entry_id": "entry-id",
            "name": "Led Zeppelin",
            "media_type": ["artist", "album", "track", "playlist"],
        }
        connection.settimeout.assert_called_once_with(15.0)
        connection.close.assert_called_once()

    @patch("ha_client.requests.post")
    def test_play_music_assistant_media_targets_ma_player(self, mock_post):
        mock_post.return_value.status_code = 200

        assert ha_client.play_music_assistant_media(
            "media_player.kitchen_sonos", "spotify://playlist/1", "playlist"
        )

        assert mock_post.call_args.kwargs["json"] == {
            "entity_id": "media_player.kitchen_sonos",
            "media_content_id": "spotify://playlist/1",
            "media_content_type": "playlist",
        }

    @patch("ha_client.requests.post", side_effect=requests.ReadTimeout)
    def test_play_music_assistant_media_read_timeout_is_pending(self, mock_post, caplog):
        assert ha_client.play_music_assistant_media(
            "media_player.kitchen_sonos", "spotify://playlist/1", "playlist"
        ) is None

        assert "may still be processing the request" in caplog.text

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

    @patch("ha_client._fetch_entity_areas", return_value={})
    @patch("requests.get")
    def test_fetch_includes_requested_conversation_entity_domains(self, mock_get, mock_areas):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"entity_id": f"{domain}.sample", "attributes": {"friendly_name": domain}, "state": "unknown"}
            for domain in ("media_player", "timer", "todo", "weather")
        ]
        mock_response.json.return_value[-1]["attributes"]["temperature_unit"] = "°C"
        mock_get.return_value = mock_response

        devices = get_ha_devices()

        assert [device["entity_id"] for device in devices] == [
            "media_player.sample", "timer.sample", "todo.sample", "weather.sample"
        ]
        assert devices[-1]["temperature_unit"] == "°C"

    @patch("ha_client.websocket.create_connection")
    def test_process_ha_conversation_returns_assistant_response(
        self, mock_create_connection, monkeypatch
    ):
        monkeypatch.setattr("ha_client.HA_URL", "http://ha.example:8123")
        monkeypatch.setattr("ha_client.HA_TOKEN", "test-token")
        connection = MagicMock()
        connection.recv.side_effect = [
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            json.dumps({
                "id": 3,
                "type": "result",
                "success": True,
                "result": {
                    "response": {
                        "response": {
                            "speech": {"plain": {"speech": "The weather is clear."}}
                        }
                    }
                },
            }),
        ]
        mock_create_connection.return_value = connection

        response = process_ha_conversation("what is the weather like")

        assert response["response"]["speech"]["plain"]["speech"] == "The weather is clear."
        request = json.loads(connection.send.call_args_list[-1].args[0])
        assert request == {
            "id": 3,
            "type": "call_service",
            "domain": "conversation",
            "service": "process",
            "service_data": {
                "text": "what is the weather like",
                "language": "en",
                "agent_id": "conversation.home_assistant",
            },
            "return_response": True,
        }

    @patch("ha_client.websocket.create_connection")
    def test_get_ha_todo_items_returns_actionable_items(
        self, mock_create_connection, monkeypatch
    ):
        monkeypatch.setattr("ha_client.HA_URL", "http://ha.example:8123")
        monkeypatch.setattr("ha_client.HA_TOKEN", "test-token")
        connection = MagicMock()
        connection.recv.side_effect = [
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            json.dumps({
                "id": 4,
                "type": "result",
                "success": True,
                "result": {
                    "response": {
                        "todo.shopping_list": {
                            "items": [
                                {"summary": "Apples", "status": "needs_action"},
                                {"summary": "Bread", "status": "needs_action"},
                            ]
                        }
                    }
                },
            }),
        ]
        mock_create_connection.return_value = connection

        items = get_ha_todo_items("todo.shopping_list")

        assert items == [
            {"summary": "Apples", "status": "needs_action"},
            {"summary": "Bread", "status": "needs_action"},
        ]
        request = json.loads(connection.send.call_args_list[-1].args[0])
        assert request["domain"] == "todo"
        assert request["service"] == "get_items"
        assert request["target"] == {"entity_id": "todo.shopping_list"}
        assert request["service_data"] == {"status": ["needs_action"]}
        assert request["return_response"] is True

    @patch("ha_client.websocket.create_connection")
    def test_get_ha_weather_forecast_calls_daily_weather_service(
        self, mock_create_connection, monkeypatch
    ):
        monkeypatch.setattr("ha_client.HA_URL", "http://ha.example:8123")
        monkeypatch.setattr("ha_client.HA_TOKEN", "test-token")
        connection = MagicMock()
        connection.recv.side_effect = [
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            json.dumps({
                "id": 5,
                "type": "result",
                "success": True,
                "result": {
                    "response": {
                        "weather.forecast_home": {
                            "forecast": [{
                                "condition": "sunny",
                                "temperature": 20,
                                "templow": 10,
                            }]
                        }
                    }
                },
            }),
        ]
        mock_create_connection.return_value = connection

        forecast = get_ha_weather_forecast("weather.forecast_home")

        assert forecast == {
            "condition": "sunny",
            "temperature": 20,
            "templow": 10,
        }
        request = json.loads(connection.send.call_args_list[-1].args[0])
        assert request["domain"] == "weather"
        assert request["service"] == "get_forecasts"
        assert request["target"] == {"entity_id": "weather.forecast_home"}
        assert request["service_data"] == {"type": "daily"}
        assert request["return_response"] is True

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