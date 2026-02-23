#!/usr/bin/env python3
"""Command-line tool for Home Assistant integration.

This script is designed to be called by Home Assistant's command_line sensor.
It reads sun position and returns whether the plant receives direct sunlight.

Usage:
    # With explicit sun position:
    python check_plant_sun.py <azimuth> <elevation> [config_path]
    python check_plant_sun.py 180 45

    # Auto-calculate sun position from config (standalone mode):
    python check_plant_sun.py --config /path/to/config.json
    python check_plant_sun.py --config /path/to/config.json --json

Returns:
    Prints "on" if plant receives sunlight, "off" otherwise.
    Exit code 0 on success.

Home Assistant Configuration Example:
    See ha_config_example.yaml in this directory.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sun_hit_detector.homeassistant.service import check_sunlight, get_sunlight_details


def get_current_sun_position(config_path: str | None = None):
    """Calculate current sun position based on location in config."""
    from sun_hit_detector.core.models import Config
    from sun_hit_detector.core.sun_position import calculate_sun_position

    config = Config.from_json_file(config_path or "config/default_config.json")
    now = datetime.now()

    if config.location is None:
        raise ValueError("Config does not include location data")

    sun_pos = calculate_sun_position(
        config.location.latitude,
        config.location.longitude,
        now,
        timezone_offset=config.location.timezone_offset,
    )
    return sun_pos.azimuth_deg, sun_pos.elevation_deg


def main():
    parser = argparse.ArgumentParser(description="Check if plant receives direct sunlight")
    parser.add_argument("azimuth", nargs="?", type=float, help="Sun azimuth angle")
    parser.add_argument("elevation", nargs="?", type=float, help="Sun elevation angle")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--windows", action="store_true",
                       help="Output window sun status instead of plant status")
    parser.add_argument("--window-id", type=str,
                       help="Check specific window (with --windows, returns on/off)")
    
    args = parser.parse_args()

    try:
        # Get sun position
        if args.azimuth is not None and args.elevation is not None:
            azimuth = args.azimuth
            elevation = args.elevation
            config_path = args.config
        elif args.config:
            # Auto-calculate sun position from config
            azimuth, elevation = get_current_sun_position(args.config)
            config_path = args.config
        else:
            # No arguments - try default config
            try:
                azimuth, elevation = get_current_sun_position()
                config_path = None
            except Exception:
                print("off")
                sys.exit(0)

        # Handle --windows flag
        if args.windows:
            from sun_hit_detector.homeassistant.service import get_window_sun_status, check_window_sunlight
            import json

            if args.json:
                # JSON output with all window details
                status = get_window_sun_status(azimuth, elevation, config_path)
                print(json.dumps(status))
            elif args.window_id:
                # Binary on/off for specific window
                is_sunny = check_window_sunlight(args.window_id, azimuth, elevation, config_path)
                print("on" if is_sunny else "off")
            else:
                # Human-readable list of windows in sun
                status = get_window_sun_status(azimuth, elevation, config_path)
                if status['windows_in_sun']:
                    print("Windows in sun:", ", ".join(status['windows_in_sun']))
                else:
                    reason = status.get('reason', 'No windows receiving sun')
                    print(reason)
            sys.exit(0)

        # Existing plant logic continues...
        if args.json:
            import json
            details = get_sunlight_details(azimuth, elevation, config_path)
            print(json.dumps(details))
        else:
            is_hit = check_sunlight(azimuth, elevation, config_path)
            print("on" if is_hit else "off")

    except Exception as e:
        if args.json:
            import json
            print(json.dumps({"is_hit": False, "error": str(e)}))
        else:
            print("off", file=sys.stdout)
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(0)  # Always exit 0 for HA compatibility


if __name__ == "__main__":
    main()
