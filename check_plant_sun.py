#!/usr/bin/env python3
"""Command-line tool for checking plant sunlight.

Uses the core sun_hit_detector library directly to determine whether
direct sunlight reaches the plant through any window.

Usage:
    python check_plant_sun.py <azimuth> <elevation> [--config path]
    python check_plant_sun.py --config config/default_config.json
    python check_plant_sun.py --config config/default_config.json --json
    python check_plant_sun.py --config config/default_config.json --windows

Returns:
    Prints "on" if plant receives sunlight, "off" otherwise.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from sun_hit_detector.core.models import Config
from sun_hit_detector.core.hit_test import (
    check_sun_hits_plant_from_config,
    check_plant_hit_per_window_from_config,
)
from sun_hit_detector.core.window_sun import check_windows_from_config
from sun_hit_detector.core.sun_position import calculate_sun_position


def get_current_sun_position(config: Config):
    """Calculate current sun position based on location in config."""
    if config.location is None:
        raise ValueError("Config does not include location data")
    now = datetime.now()
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
    parser.add_argument("--config", type=str, default="config/default_config.json",
                       help="Path to config JSON file")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--windows", action="store_true",
                       help="Output window sun status instead of plant status")

    args = parser.parse_args()

    try:
        config = Config.from_json_file(args.config)

        if args.azimuth is not None and args.elevation is not None:
            azimuth, elevation = args.azimuth, args.elevation
        else:
            azimuth, elevation = get_current_sun_position(config)

        if args.windows:
            window_result = check_windows_from_config(azimuth, elevation, config)
            plant_hits = check_plant_hit_per_window_from_config(azimuth, elevation, config)
            if args.json:
                output = window_result.to_dict()
                output["plant_hits_per_window"] = plant_hits
                print(json.dumps(output))
            else:
                for wid, detail in window_result.window_details.items():
                    plant_hit = "→ HITS PLANT" if plant_hits.get(wid) else ""
                    sun = "☀" if detail.is_in_sun else "·"
                    print(f"  {sun} {wid}: intensity={detail.intensity_factor:.2f} "
                          f"angle={detail.sun_angle_to_normal_deg:.0f}° {plant_hit}")
            sys.exit(0)

        result = check_sun_hits_plant_from_config(azimuth, elevation, config)
        if args.json:
            print(json.dumps({
                "is_hit": result.is_hit,
                "window_id": result.window_id,
                "reason": result.reason,
                "sun_azimuth": azimuth,
                "sun_elevation": elevation,
            }))
        else:
            print("on" if result.is_hit else "off")

    except Exception as e:
        if args.json:
            print(json.dumps({"is_hit": False, "error": str(e)}))
        else:
            print("off", file=sys.stdout)
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
