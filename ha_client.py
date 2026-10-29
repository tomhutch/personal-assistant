import json
import logging
import time

import requests
import websocket
from config import HA_URL, HA_TOKEN

logger = logging.getLogger(__name__)
_AREA_CACHE_TTL_SECONDS = 300
_area_cache = {}
_area_cache_expires_at = 0.0


def _websocket_result(connection, request_id):
    while True:
        message = json.loads(connection.recv())
        if message.get("id") == request_id:
            if not message.get("success"):
                raise RuntimeError(message.get("error", "Registry request failed"))
            return message.get("result") or []


def _fetch_entity_areas():
    global _area_cache, _area_cache_expires_at

    if _area_cache_expires_at > time.monotonic():
        return _area_cache
    if not HA_TOKEN:
        return {}

    websocket_url = HA_URL.rstrip("/")
    if websocket_url.startswith("https://"):
        websocket_url = "wss://" + websocket_url[len("https://"):]
    elif websocket_url.startswith("http://"):
        websocket_url = "ws://" + websocket_url[len("http://"):]
    else:
        logger.debug("Unsupported Home Assistant URL scheme for area lookup.")
        return {}

    connection = None
    try:
        connection = websocket.create_connection(
            f"{websocket_url}/api/websocket", timeout=2.0
        )
        greeting = json.loads(connection.recv())
        if greeting.get("type") != "auth_required":
            raise RuntimeError("Unexpected Home Assistant WebSocket greeting")

        connection.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
        auth_result = json.loads(connection.recv())
        if auth_result.get("type") != "auth_ok":
            raise RuntimeError("Home Assistant WebSocket authentication failed")

        connection.send(json.dumps({"id": 1, "type": "config/entity_registry/list"}))
        entities = _websocket_result(connection, 1)
        connection.send(json.dumps({"id": 2, "type": "config/area_registry/list"}))
        areas = _websocket_result(connection, 2)

        area_names = {area["area_id"]: area["name"] for area in areas if area.get("name")}
        _area_cache = {
            entity["entity_id"]: area_names[entity["area_id"]]
            for entity in entities
            if entity.get("entity_id") and entity.get("area_id") in area_names
        }
        _area_cache_expires_at = time.monotonic() + _AREA_CACHE_TTL_SECONDS
        return _area_cache
    except Exception as e:
        logger.debug("Could not load Home Assistant area registry: %s", e)
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
        is_device = entity_id.startswith(("switch.", "light."))
        is_controllable_group = entity_id.startswith("group.") and any(
            member.startswith(("switch.", "light.")) for member in members
        )
        if is_device or is_controllable_group:
            device = {
                "entity_id": entity_id,
                "name": name,
                "state": state.get("state")
            }
            if is_controllable_group:
                device["members"] = members
            if area := entity_areas.get(entity_id):
                device["area"] = area
            controllable.append(device)

    device_summary = [f"{d['name']} ({d['entity_id']}): {d['state']}" for d in controllable]
    logger.debug("➜ Discovered %d controllable device(s): %s", len(controllable), device_summary)
    return controllable

# Alias for backward compatibility if needed
get_ha_devices = fetch_ha_devices

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