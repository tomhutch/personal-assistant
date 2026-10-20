import logging
import requests
from config import HA_URL, HA_TOKEN

logger = logging.getLogger(__name__)

def fetch_ha_devices():
    """Fetches all states from Home Assistant and filters for controllable switches/lights."""
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    
    try:
        response = requests.get(f"{HA_URL}/api/states", headers=headers, timeout=5.0)
        if response.status_code != 200:
            logger.error("❌ Failed to fetch HA states. Status: %d", response.status_code)
            return []
            
        states = response.json()
        controllable = []
        
        for state in states:
            entity_id = state.get("entity_id", "")
            if entity_id.startswith("switch.") or entity_id.startswith("light."):
                controllable.append({
                    "entity_id": entity_id,
                    "name": state.get("attributes", {}).get("friendly_name", entity_id),
                    "state": state.get("state")
                })
                
        logger.info("➜ Discovered %d controllable device(s).", len(controllable))
        return controllable
        
    except requests.RequestException as e:
        logger.error("❌ Error connecting to Home Assistant: %s", e)
        return []

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
        logger.info("  ➜ HA Response: %s", response.json() if response.content else "[]")
        
        return response.status_code == 200
    except requests.RequestException as e:
        logger.error("❌ Failed to execute Home Assistant command: %s", e)
        return False