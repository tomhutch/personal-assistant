import logging
import requests
from config import HA_URL, HA_TOKEN

logger = logging.getLogger(__name__)

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

    controllable = []

    for state in states:
        entity_id = state.get("entity_id", "")
        name = state.get("attributes", {}).get("friendly_name", entity_id)
        if "network_led" in entity_id.casefold() or "network_led" in name.casefold():
            continue
        if entity_id.startswith("switch.") or entity_id.startswith("light."):
            controllable.append({
                "entity_id": entity_id,
                "name": name,
                "state": state.get("state")
            })

    device_summary = [f"{d['name']} ({d['entity_id']}): {d['state']}" for d in controllable]
    logger.debug("➜ Discovered %d controllable device(s): %s", len(controllable), device_summary)
    return controllable

# Alias for backward compatibility if needed
get_ha_devices = fetch_ha_devices

def execute_ha_command(domain: str, service: str, entity_id: str) -> bool:
    """Executes a service call on a Home Assistant entity (e.g., switch/turn_off)."""
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"entity_id": entity_id}
    
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