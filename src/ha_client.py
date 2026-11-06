import json
import logging
import time

import requests
import websocket
from config import HA_URL, HA_TOKEN

logger = logging.getLogger(__name__)
_AREA_CACHE_TTL_SECONDS = 300
_area_cache = {}
_entity_floors = {}
_entity_platforms = {}
_entity_config_entry_ids = {}
_area_cache_expires_at = 0.0


def _websocket_result(connection, request_id):
    while True:
        message = json.loads(connection.recv())
        if message.get("id") == request_id:
            if not message.get("success"):
                raise RuntimeError(message.get("error", "Registry request failed"))
            return message.get("result") or []


def _create_authenticated_websocket():
    websocket_url = HA_URL.rstrip("/")
    if websocket_url.startswith("https://"):
        websocket_url = "wss://" + websocket_url[len("https://"):]
    elif websocket_url.startswith("http://"):
        websocket_url = "ws://" + websocket_url[len("http://"):]
    else:
        raise ValueError("Unsupported Home Assistant URL scheme")

    connection = websocket.create_connection(
        f"{websocket_url}/api/websocket", timeout=2.0
    )
    greeting = json.loads(connection.recv())
    if greeting.get("type") != "auth_required":
        connection.close()
        raise RuntimeError("Unexpected Home Assistant WebSocket greeting")

    connection.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
    auth_result = json.loads(connection.recv())
    if auth_result.get("type") != "auth_ok":
        connection.close()
        raise RuntimeError("Home Assistant WebSocket authentication failed")

    return connection


def process_ha_conversation(text: str, language: str = "en") -> dict | None:
    """Ask Home Assistant to handle a recognized built-in intent and return its response."""
    if not HA_TOKEN:
        logger.error("Cannot process Home Assistant conversation without an access token.")
        return None

    connection = None
    try:
        connection = _create_authenticated_websocket()
        connection.send(json.dumps({
            "id": 3,
            "type": "call_service",
            "domain": "conversation",
            "service": "process",
            "service_data": {
                "text": text,
                "language": language,
                "agent_id": "conversation.home_assistant",
            },
            "return_response": True,
        }))
        result = _websocket_result(connection, 3)
        response = result.get("response") if isinstance(result, dict) else None
        if not isinstance(response, dict):
            logger.error("Home Assistant conversation returned no response data.")
            return None
        return response
    except Exception as exc:
        logger.error("Failed to process Home Assistant conversation: %s", exc)
        return None
    finally:
        if connection is not None:
            connection.close()


def get_ha_todo_items(entity_id: str) -> list[dict] | None:
    """Return actionable items from a Home Assistant todo entity."""
    if not entity_id.startswith("todo."):
        logger.error("Refusing to fetch todo items for non-todo entity %s.", entity_id)
        return None
    if not HA_TOKEN:
        logger.error("Cannot fetch todo items without a Home Assistant access token.")
        return None

    connection = None
    try:
        connection = _create_authenticated_websocket()
        connection.send(json.dumps({
            "id": 4,
            "type": "call_service",
            "domain": "todo",
            "service": "get_items",
            "target": {"entity_id": entity_id},
            "service_data": {"status": ["needs_action"]},
            "return_response": True,
        }))
        result = _websocket_result(connection, 4)
        response = result.get("response", {}) if isinstance(result, dict) else {}
        todo_list = response.get(entity_id, {}) if isinstance(response, dict) else {}
        items = todo_list.get("items", []) if isinstance(todo_list, dict) else []
        return [item for item in items if isinstance(item, dict)]
    except Exception as exc:
        logger.error("Failed to fetch Home Assistant todo items for %s: %s", entity_id, exc)
        return None
    finally:
        if connection is not None:
            connection.close()


def get_ha_weather_forecast(entity_id: str) -> dict | None:
    """Return today's forecast from a Home Assistant weather entity."""
    if not entity_id.startswith("weather."):
        logger.error("Refusing to fetch weather forecast for non-weather entity %s.", entity_id)
        return None
    if not HA_TOKEN:
        logger.error("Cannot fetch weather forecast without a Home Assistant access token.")
        return None

    connection = None
    try:
        connection = _create_authenticated_websocket()
        connection.send(json.dumps({
            "id": 5,
            "type": "call_service",
            "domain": "weather",
            "service": "get_forecasts",
            "target": {"entity_id": entity_id},
            "service_data": {"type": "daily"},
            "return_response": True,
        }))
        result = _websocket_result(connection, 5)
        response = result.get("response", {}) if isinstance(result, dict) else {}
        weather_data = response.get(entity_id, {}) if isinstance(response, dict) else {}
        forecasts = weather_data.get("forecast", []) if isinstance(weather_data, dict) else []
        return forecasts[0] if forecasts and isinstance(forecasts[0], dict) else None
    except Exception as exc:
        logger.error("Failed to fetch Home Assistant weather forecast for %s: %s", entity_id, exc)
        return None
    finally:
        if connection is not None:
            connection.close()


def _fetch_entity_areas():
    global _area_cache, _entity_floors, _entity_platforms
    global _entity_config_entry_ids, _area_cache_expires_at

    if _area_cache_expires_at > time.monotonic():
        return _area_cache
    if not HA_TOKEN:
        _entity_floors = {}
        _entity_platforms = {}
        _entity_config_entry_ids = {}
        return {}

    connection = None
    try:
        connection = _create_authenticated_websocket()
        connection.send(json.dumps({"id": 1, "type": "config/entity_registry/list"}))
        entities = _websocket_result(connection, 1)
        connection.send(json.dumps({"id": 2, "type": "config/area_registry/list"}))
        areas = _websocket_result(connection, 2)
        floors = []
        try:
            connection.send(json.dumps({"id": 3, "type": "config/floor_registry/list"}))
            floors = _websocket_result(connection, 3)
        except Exception as e:
            logger.debug("Could not load Home Assistant floor registry: %s", e)

        area_names = {area["area_id"]: area["name"] for area in areas if area.get("name")}
        floor_names = {floor["floor_id"]: floor["name"] for floor in floors if floor.get("name")}
        area_floors = {
            area["area_id"]: floor_names[area["floor_id"]]
            for area in areas
            if area.get("area_id") and area.get("floor_id") in floor_names
        }
        _area_cache = {
            entity["entity_id"]: area_names[entity["area_id"]]
            for entity in entities
            if entity.get("entity_id") and entity.get("area_id") in area_names
        }
        _entity_floors = {
            entity["entity_id"]: area_floors[entity["area_id"]]
            for entity in entities
            if entity.get("entity_id") and entity.get("area_id") in area_floors
        }
        _entity_platforms = {
            entity["entity_id"]: entity["platform"]
            for entity in entities
            if entity.get("entity_id") and entity.get("platform")
        }
        _entity_config_entry_ids = {
            entity["entity_id"]: entity["config_entry_id"]
            for entity in entities
            if entity.get("entity_id") and entity.get("config_entry_id")
        }
        _area_cache_expires_at = time.monotonic() + _AREA_CACHE_TTL_SECONDS
        return _area_cache
    except Exception as e:
        logger.debug("Could not load Home Assistant area registry: %s", e)
        _entity_floors = {}
        _entity_platforms = {}
        _entity_config_entry_ids = {}
        return {}
    finally:
        if connection is not None:
            connection.close()
def fetch_ha_devices():
    """Fetch controllable devices, returning None when the request fails."""
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    
    try:
        response = requests.get(f"{HA_URL}/api/states", headers=headers, timeout=5.0)
    except requests.RequestException as e:
        logger.error("❌ Error connecting to Home Assistant: %s", e)
        return None

    if response.status_code != 200:
        logger.error("❌ Failed to fetch HA states. Status: %d", response.status_code)
        return None

    try:
        states = response.json()
    except ValueError as e:
        logger.error("❌ Home Assistant returned invalid JSON for device states: %s", e)
        return None

    entity_areas = _fetch_entity_areas()
    controllable = []

    for state in states:
        entity_id = state.get("entity_id", "")
        attributes = state.get("attributes", {})
        name = attributes.get("friendly_name", entity_id)
        if "network_led" in entity_id.casefold() or "network_led" in name.casefold():
            continue
        members = attributes.get("entity_id", [])
        is_device = entity_id.startswith((
            "switch.", "light.", "media_player.", "timer.", "todo.", "weather."
        ))
        is_controllable_group = entity_id.startswith("group.") and any(
            member.startswith(("switch.", "light.")) for member in members
        )
        if is_device or is_controllable_group:
            device = {
                "entity_id": entity_id,
                "name": name,
                "state": state.get("state")
            }
            if platform := _entity_platforms.get(entity_id):
                device["platform"] = platform
            if config_entry_id := _entity_config_entry_ids.get(entity_id):
                device["config_entry_id"] = config_entry_id
            if is_controllable_group:
                device["members"] = members
            if area := entity_areas.get(entity_id):
                device["area"] = area
            if floor := _entity_floors.get(entity_id):
                device["floor"] = floor
            if entity_id.startswith("weather."):
                device["temperature_unit"] = attributes.get("temperature_unit")
            controllable.append(device)

    device_summary = [f"{d['name']} ({d['entity_id']}): {d['state']}" for d in controllable]
    logger.debug("➜ Discovered %d controllable device(s): %s", len(controllable), device_summary)
    return controllable

# Alias for backward compatibility if needed
get_ha_devices = fetch_ha_devices

class MusicAssistantError(Exception):
    """A Music Assistant failure with a message suitable for voice feedback."""


def search_music_assistant(
    query: str, config_entry_id: str, media_type: str | None = None
) -> dict:
    """Search the configured Music Assistant provider catalog through Home Assistant."""
    if not HA_TOKEN:
        raise MusicAssistantError("Home Assistant isn't configured with an access token.")
    if not config_entry_id:
        raise MusicAssistantError(
            "The Music Assistant player is missing its Home Assistant integration entry."
        )
    if media_type not in {None, "artist", "album", "track", "playlist"}:
        raise MusicAssistantError(f"Unsupported music search type: {media_type}.")

    media_types = (
        ["artist", "playlist"] if media_type == "artist"
        else [media_type] if media_type
        else ["artist", "album", "track", "playlist"]
    )

    connection = None
    try:
        connection = _create_authenticated_websocket()
        connection.settimeout(15.0)
        connection.send(json.dumps({
            "id": 6,
            "type": "call_service",
            "domain": "music_assistant",
            "service": "search",
            "service_data": {
                "config_entry_id": config_entry_id,
                "name": query,
                "media_type": media_types,
            },
            "return_response": True,
        }))
        result = _websocket_result(connection, 6)
        response = result.get("response") if isinstance(result, dict) else None
        if not isinstance(response, dict):
            raise RuntimeError("Music Assistant returned no search results.")
        return response
    except Exception as exc:
        logger.error("Music Assistant search failed for %r: %s", query, exc)
        raise MusicAssistantError("I couldn't search Music Assistant right now.") from exc
    finally:
        if connection is not None:
            connection.close()


def play_music_assistant_media(
    entity_id: str, media_id: str, media_type: str
) -> bool | None:
    """Play a Music Assistant search result on its Home Assistant media-player entity."""
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "entity_id": entity_id,
        "media_content_id": media_id,
        "media_content_type": media_type,
    }
    try:
        response = requests.post(
            f"{HA_URL}/api/services/media_player/play_media",
            headers=headers,
            json=payload,
            timeout=10.0,
        )
    except requests.ReadTimeout as e:
        logger.warning(
            "Timed out waiting for Music Assistant playback on %s; Home Assistant "
            "may still be processing the request: %s",
            entity_id,
            e,
        )
        return None
    except requests.RequestException as e:
        logger.error("❌ Failed to call Music Assistant media playback: %s", e)
        return False
    if response.status_code != 200:
        logger.error(
            "❌ Music Assistant playback failed for %s (HTTP %d): %s",
            entity_id,
            response.status_code,
            response.text[:300],
        )
        return False
    return True

def control_music_assistant_player(entity_id: str, service: str) -> bool | None:
    """Call a media control; a read timeout means the result is unknown."""
    if not entity_id.startswith("media_player.") or service not in {
        "media_pause", "media_play", "media_next_track", "media_previous_track",
    }:
        logger.error("Invalid Music Assistant control %s for %s.", service, entity_id)
        return False
    try:
        response = requests.post(
            f"{HA_URL}/api/services/media_player/{service}",
            headers={"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"},
            json={"entity_id": entity_id},
            timeout=10.0,
        )
    except requests.ReadTimeout as exc:
        logger.warning("Music Assistant control %s on %s timed out; result unknown: %s", service, entity_id, exc)
        return None
    except requests.RequestException as exc:
        logger.error("Music Assistant control %s on %s failed: %s", service, entity_id, exc)
        return False
    if response.status_code != 200:
        logger.error("Music Assistant control %s on %s failed (HTTP %d): %s", service, entity_id, response.status_code, response.text[:300])
        return False
    return True


def execute_ha_command(
    domain: str, service: str, entity_id: str, brightness_pct: float | None = None
) -> bool:
    """Executes a service call on a Home Assistant entity (e.g., switch/turn_off)."""
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"entity_id": entity_id}
    if brightness_pct is not None:
        if domain != "light" or service != "turn_on" or not 1 <= brightness_pct <= 100:
            logger.error("❌ Invalid brightness request for %s.", entity_id)
            return False
        payload["brightness_pct"] = brightness_pct
    
    try:
        url = f"{HA_URL}/api/services/{domain}/{service}"
        logger.info("🚀 Firing POST to Home Assistant for %s...", entity_id)
        response = requests.post(url, headers=headers, json=payload, timeout=5.0)

        logger.info("  ➜ HA API Status Code: %d", response.status_code)
        result = response.json() if response.content else []
        logger.info("  ➜ HA Response: %s", result)

        if response.status_code != 200:
            return False

        entity_changed = isinstance(result, list) and any(
            isinstance(state, dict) and state.get("entity_id") == entity_id
            for state in result
        )
        if (
            not entity_changed
            and domain == "homeassistant"
            and entity_id.startswith("group.")
            and isinstance(result, list)
            and any(isinstance(state, dict) for state in result)
        ):
            entity_changed = True
        if not entity_changed:
            logger.error(
                "❌ Home Assistant returned HTTP 200 but reported no state change for %s.",
                entity_id,
            )
            return False

        return True
    except requests.RequestException as e:
        logger.error("❌ Failed to execute Home Assistant command: %s", e)
        return False
