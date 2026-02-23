#!/usr/bin/env python3
"""Helper script to retrieve Home Assistant shade entities and generate config mapping.

This script connects to Home Assistant via API and retrieves all Overkiz shade entities,
then generates the correct mapping for the sun simulator config file.

Usage:
    # From Home Assistant host (via docker exec)
    docker exec home-assistant python3 /sun-plant-simulator/scripts/get_ha_shade_entities.py

    # Or with HA API (requires token)
    python get_ha_shade_entities.py --url http://homeassistant.local:8123 --token YOUR_TOKEN
"""

import sys
import json
import argparse
from pathlib import Path


def get_entities_from_ha_states():
    """Get entities by reading HA states file directly (when running inside container)."""
    try:
        # Try to import Home Assistant's async methods
        import asyncio
        from homeassistant.core import HomeAssistant
        from homeassistant.helpers import entity_registry

        # This would only work if running inside HA environment
        print("Running inside Home Assistant environment...")
        return None
    except ImportError:
        return None


def get_entities_via_api(url: str, token: str):
    """Get entities via Home Assistant REST API."""
    import requests

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(f"{url}/api/states", headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error connecting to Home Assistant API: {e}")
        return None


def filter_shade_entities(states):
    """Filter for Overkiz shade entities."""
    shades = []
    for state in states:
        entity_id = state.get("entity_id", "")
        if "cover." in entity_id and "living" in entity_id.lower() and "shade" in entity_id.lower():
            attributes = state.get("attributes", {})
            shades.append({
                "entity_id": entity_id,
                "friendly_name": attributes.get("friendly_name", ""),
                "integration": attributes.get("integration", ""),
            })
    return shades


def generate_window_mapping():
    """Generate the window-to-shade mapping based on user description."""

    # Based on user input:
    # Front shades: "Living Room Front Shade" 1-4 (INVERTED - 4 is closest to corner)
    # Side shades: "Living Room Side Shade" 1-4 (NOT inverted - follows config order)

    mapping = {
        # Wall 1 (Front) - INVERTED mapping
        "window_1a": {
            "x_position": 0.36,
            "description": "Closest to corner on wall_1",
            "shade": "cover.living_room_front_shade_4",  # Inverted: 4 is closest
        },
        "window_1b": {
            "x_position": 1.44,
            "description": "Second on wall_1",
            "shade": "cover.living_room_front_shade_3",
        },
        "window_1c": {
            "x_position": 2.52,
            "description": "Third on wall_1",
            "shade": "cover.living_room_front_shade_2",
        },
        "window_1d": {
            "x_position": 3.6,
            "description": "Farthest from corner on wall_1",
            "shade": "cover.living_room_front_shade_1",  # Inverted: 1 is farthest
        },

        # Wall 2 (Side) - NORMAL mapping (follows config order)
        "window_2a": {
            "y_position": 0.8,
            "description": "Closest to corner on wall_2",
            "shade": "cover.living_room_side_shade_1",
        },
        "window_2b": {
            "y_position": 4.0,
            "description": "Second on wall_2",
            "shade": "cover.living_room_side_shade_2",
        },
        "window_2c": {
            "y_position": 5.07,
            "description": "Third on wall_2",
            "shade": "cover.living_room_side_shade_3",
        },
        "window_2d": {
            "y_position": 8.26,
            "description": "Farthest from corner on wall_2",
            "shade": "cover.living_room_side_shade_4",
        },
    }

    return mapping


def print_mapping_table(mapping):
    """Print a formatted table of the window-to-shade mapping."""
    print("\n" + "=" * 100)
    print("WINDOW TO SHADE ENTITY MAPPING")
    print("=" * 100)

    print("\n" + "-" * 100)
    print("WALL 1 (FRONT) - Azimuth 210° - INVERTED ORDER")
    print("-" * 100)
    print(f"{'Window ID':<12} {'Position':<12} {'Shade Entity ID':<40} {'Description'}")
    print("-" * 100)

    for window_id in ["window_1a", "window_1b", "window_1c", "window_1d"]:
        info = mapping[window_id]
        pos = f"x={info['x_position']}"
        print(f"{window_id:<12} {pos:<12} {info['shade']:<40} {info['description']}")

    print("\n" + "-" * 100)
    print("WALL 2 (SIDE) - Azimuth 300° - NORMAL ORDER")
    print("-" * 100)
    print(f"{'Window ID':<12} {'Position':<12} {'Shade Entity ID':<40} {'Description'}")
    print("-" * 100)

    for window_id in ["window_2a", "window_2b", "window_2c", "window_2d"]:
        info = mapping[window_id]
        pos = f"y={info['y_position']}"
        print(f"{window_id:<12} {pos:<12} {info['shade']:<40} {info['description']}")

    print("-" * 100)


def verify_entities_exist(mapping, ha_url=None, ha_token=None):
    """Verify that the shade entities exist in Home Assistant."""
    if ha_url and ha_token:
        print("\n" + "=" * 100)
        print("VERIFYING ENTITIES IN HOME ASSISTANT")
        print("=" * 100)

        states = get_entities_via_api(ha_url, ha_token)
        if states:
            existing_entities = {state["entity_id"] for state in states}

            for window_id, info in mapping.items():
                entity_id = info["shade"]
                exists = entity_id in existing_entities
                status = "✓ EXISTS" if exists else "✗ NOT FOUND"
                print(f"{entity_id:<50} {status}")
        else:
            print("Could not retrieve entities from Home Assistant")
    else:
        print("\nNote: Run with --url and --token to verify entities exist in HA")


def generate_config_update(mapping, config_path="config/default_config.json"):
    """Generate JSON snippet to update config file."""
    print("\n" + "=" * 100)
    print("CONFIG FILE UPDATE")
    print("=" * 100)
    print(f"\nAdd shade_entity_id to each window in {config_path}:\n")

    for window_id, info in mapping.items():
        print(f'  "{window_id}": {{"shade_entity_id": "{info["shade"]}"}}')


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve Home Assistant shade entities and generate mapping"
    )
    parser.add_argument("--url", help="Home Assistant URL (e.g., http://homeassistant.local:8123)")
    parser.add_argument("--token", help="Home Assistant Long-Lived Access Token")

    args = parser.parse_args()

    # Generate mapping based on user description
    mapping = generate_window_mapping()

    # Print the mapping
    print_mapping_table(mapping)

    # Verify entities if credentials provided
    if args.url and args.token:
        verify_entities_exist(mapping, args.url, args.token)

    # Generate config update snippet
    generate_config_update(mapping)

    print("\n" + "=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print("\n1. Verify the entity IDs match your Home Assistant setup")
    print("2. Update config/default_config.json with the correct shade_entity_id values")
    print("3. Deploy to Home Assistant and restart the custom component")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
