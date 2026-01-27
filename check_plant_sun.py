#!/usr/bin/env python3
"""Command-line tool for Home Assistant integration.

This script is designed to be called by Home Assistant's command_line sensor.
It reads sun position and returns whether the plant receives direct sunlight.

Usage:
    python check_plant_sun.py <azimuth> <elevation>
    python check_plant_sun.py 180 45

Returns:
    Prints "on" if plant receives sunlight, "off" otherwise.
    Exit code 0 on success, 1 on error.

Home Assistant Configuration Example:
    See ha_config_example.yaml in this directory.
"""

import sys
from pathlib import Path

# Add project to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sun_plant_simulator.homeassistant.service import check_sunlight, get_sunlight_details


def main():
    # Check arguments
    if len(sys.argv) < 3:
        print("off")  # Default to off if no arguments
        sys.exit(0)

    try:
        azimuth = float(sys.argv[1])
        elevation = float(sys.argv[2])
    except ValueError:
        print("off")
        sys.exit(1)

    # Optional: config path as third argument
    config_path = sys.argv[3] if len(sys.argv) > 3 else None

    # Check if plant receives sunlight
    try:
        is_hit = check_sunlight(azimuth, elevation, config_path)
        print("on" if is_hit else "off")
    except Exception as e:
        print(f"off", file=sys.stdout)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main_json():
    """Alternative entry point that returns JSON for more details."""
    if len(sys.argv) < 3:
        print('{"is_hit": false, "error": "missing arguments"}')
        sys.exit(0)

    try:
        azimuth = float(sys.argv[1])
        elevation = float(sys.argv[2])
        config_path = sys.argv[3] if len(sys.argv) > 3 else None

        import json
        details = get_sunlight_details(azimuth, elevation, config_path)
        print(json.dumps(details))
    except Exception as e:
        print(f'{{"is_hit": false, "error": "{e}"}}')
        sys.exit(1)


if __name__ == "__main__":
    # Use --json flag for JSON output
    if "--json" in sys.argv:
        sys.argv.remove("--json")
        main_json()
    else:
        main()
