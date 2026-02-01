#!/usr/bin/env python3
"""Generate the interactive visualization from current config.

Run this script after editing config/default_config.json to update the visualization.

Usage:
    python generate_visualization.py                    # Use today's date
    python generate_visualization.py 2024-06-21        # Use specific date (summer solstice)
    python generate_visualization.py 2024-12-21        # Use specific date (winter solstice)
"""

import json
import sys
from datetime import date, datetime
from pathlib import Path

from sun_plant_simulator.core.models import Config
from sun_plant_simulator.core.sun_position import (
    generate_sun_data_for_date,
    get_sunrise_sunset,
    resolve_timezone_offset,
)
from sun_plant_simulator.visualization.interactive import create_time_slider_visualization


def _format_utc_offset(hours: float) -> str:
    """Format a UTC offset in hours as UTC±HH:MM."""
    sign = "+" if hours >= 0 else "-"
    total_minutes = int(round(abs(hours) * 60))
    hh, mm = divmod(total_minutes, 60)
    return f"UTC{sign}{hh:02d}:{mm:02d}"


def main():
    config_path = Path("config/default_config.json")
    output_path = Path("examples/interactive_simulation.html")

    # Parse date from command line or use today
    if len(sys.argv) > 1:
        try:
            target_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            print(f"Invalid date format: {sys.argv[1]}")
            print("Use YYYY-MM-DD format, e.g., 2024-06-21")
            sys.exit(1)
    else:
        target_date = date.today()

    print("=" * 60)
    print("Sun-Plant Visualization Generator")
    print("=" * 60)

    # Load config
    print("\nLoading config...")
    config = Config.from_json_file(config_path)

    # Get location from config
    with open(config_path) as f:
        raw_config = json.load(f)

    location = raw_config.get("location", {})
    latitude = location.get("latitude", 28.349035)
    longitude = location.get("longitude", -81.245962)
    timezone_offset = location.get("timezone_offset", -5)
    timezone_name = location.get("timezone_name")

    noon_dt = datetime(target_date.year, target_date.month, target_date.day, 12, 0)
    active_offset = resolve_timezone_offset(noon_dt, timezone_offset, timezone_name)

    print(f"\nLocation: {latitude:.4f}°N, {longitude:.4f}°W")
    offset_label = _format_utc_offset(active_offset)
    if timezone_name:
        print(f"Timezone: {timezone_name} ({offset_label})")
    else:
        print(f"Timezone offset: {offset_label}")
    print(f"Date: {target_date.strftime('%Y-%m-%d (%A)')}")

    # Get sunrise/sunset
    sunrise, sunset = get_sunrise_sunset(
        latitude,
        longitude,
        target_date,
        timezone_offset=timezone_offset,
        timezone_name=timezone_name,
    )
    if sunrise and sunset:
        print(f"Sunrise: ~{sunrise.strftime('%H:%M')}")
        print(f"Sunset: ~{sunset.strftime('%H:%M')}")

    # Generate sun positions for the day (5-minute intervals)
    print("\nCalculating sun positions...")
    sun_data = generate_sun_data_for_date(
        latitude=latitude,
        longitude=longitude,
        target_date=target_date,
        timezone_offset=timezone_offset,
        timezone_name=timezone_name,
        interval_minutes=5,
    )
    print(f"  Generated {len(sun_data)} time points")

    # Show some sample positions
    print("\n  Sample positions:")
    for i in [0, len(sun_data)//4, len(sun_data)//2, 3*len(sun_data)//4, -1]:
        if i < len(sun_data):
            d = sun_data[i]
            print(f"    {d['timestamp']}: Az={d['azimuth_deg']:.0f}°, El={d['elevation_deg']:.0f}°")

    # Show plant and window info
    print(f"\nPlant position: x={config.plant.center_x}, y={config.plant.center_y}")
    print(f"Windows: {len(config.windows)}")

    # Generate visualization
    print("\nGenerating visualization...")
    create_time_slider_visualization(
        config=config,
        sun_data=sun_data,
        output_path=str(output_path),
        date_str=target_date.strftime("%Y-%m-%d"),
    )

    print(f"\n{'=' * 60}")
    print(f"Saved to: {output_path.absolute()}")
    print(f"Date: {target_date.strftime('%Y-%m-%d')}")
    print("Open this file in your browser (Ctrl+F5 to refresh)")
    print("=" * 60)


if __name__ == "__main__":
    main()
