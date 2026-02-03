#!/usr/bin/env python3
"""Local testing script for window sun exposure feature.

This script allows testing the window sun detection functionality locally
before deploying to Home Assistant. Supports both single position tests
and time-range tests throughout the day.

Usage:
    # Test current time
    python test_window_sun.py

    # Test specific sun position
    python test_window_sun.py --azimuth 210 --elevation 30

    # Test time range throughout day (6am-8pm)
    python test_window_sun.py --time-range

    # Use custom config
    python test_window_sun.py --config /path/to/config.json --time-range
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sun_plant_simulator.core.models import Config
from sun_plant_simulator.core.window_sun import check_windows_from_config
from sun_plant_simulator.core.sun_position import calculate_sun_position


def print_window_details(result: dict, verbose: bool = False):
    """Print detailed window sun exposure information.

    Args:
        result: WindowSunResult dictionary from check_windows_from_config
        verbose: If True, show all windows; if False, only show windows in sun
    """
    print("\n" + "=" * 80)
    print(f"Sun Position: Azimuth={result['sun_azimuth_deg']:.1f}°, "
          f"Elevation={result['sun_elevation_deg']:.1f}°")
    print("=" * 80)

    if result.get('reason'):
        print(f"\nReason: {result['reason']}")

    print(f"\nWindows in sun: {len(result['windows_in_sun'])}")
    if result['windows_in_sun']:
        print(f"  {', '.join(result['windows_in_sun'])}")

    print("\nWindow Details:")
    print("-" * 80)
    print(f"{'Window ID':<15} {'In Sun':<10} {'Angle (deg)':<12} {'Intensity':<10}")
    print("-" * 80)

    for window_id, detail in result['window_details'].items():
        is_in_sun = detail['is_in_sun']

        # Skip windows not in sun if not verbose
        if not verbose and not is_in_sun:
            continue

        in_sun_str = "YES" if is_in_sun else "no"
        angle = detail['sun_angle_to_normal_deg']
        intensity = detail['intensity_factor']

        print(f"{window_id:<15} {in_sun_str:<10} {angle:<12.1f} {intensity:<10.3f}")

    print("-" * 80)


def print_time_range_table(time_results: list[tuple]):
    """Print table of sun positions and windows in sun over time.

    Args:
        time_results: List of (time, azimuth, elevation, windows_in_sun) tuples
    """
    print("\n" + "=" * 100)
    print("Time Range Test: Window Sun Exposure Throughout Day")
    print("=" * 100)
    print(f"{'Time':<10} {'Azimuth':<10} {'Elevation':<12} {'Windows in Sun'}")
    print("-" * 100)

    for time_str, azimuth, elevation, windows_in_sun in time_results:
        windows_str = ", ".join(windows_in_sun) if windows_in_sun else "(none)"
        print(f"{time_str:<10} {azimuth:<10.1f} {elevation:<12.1f} {windows_str}")

    print("-" * 100)


def test_single_position(azimuth: float, elevation: float, config: Config):
    """Test a single sun position.

    Args:
        azimuth: Sun azimuth in degrees
        elevation: Sun elevation in degrees
        config: Configuration object
    """
    result = check_windows_from_config(azimuth, elevation, config)
    print_window_details(result.to_dict(), verbose=True)


def test_time_range(config: Config, date: datetime = None):
    """Test window sun exposure over a time range (6am-8pm).

    Args:
        config: Configuration object
        date: Date to test (default: today)
    """
    if config.location is None:
        print("Error: Config must include location data for time range test")
        sys.exit(1)

    if date is None:
        date = datetime.now()

    # Test every hour from 6am to 8pm
    time_results = []

    for hour in range(6, 21):  # 6am to 8pm
        test_time = date.replace(hour=hour, minute=0, second=0, microsecond=0)

        # Calculate sun position for this time
        sun_pos = calculate_sun_position(
            config.location.latitude,
            config.location.longitude,
            test_time,
            timezone_offset=config.location.timezone_offset,
        )

        azimuth = sun_pos.azimuth_deg
        elevation = sun_pos.elevation_deg

        # Check windows at this sun position
        result = check_windows_from_config(azimuth, elevation, config)

        time_str = test_time.strftime("%I:%M %p")
        time_results.append((
            time_str,
            azimuth,
            elevation,
            result.windows_in_sun,
        ))

    # Print results table
    print_time_range_table(time_results)

    # Print summary by window
    print("\n" + "=" * 100)
    print("Summary: Hours of Sun Exposure by Window")
    print("=" * 100)

    # Count hours for each window
    window_hours = {}
    for _, _, _, windows_in_sun in time_results:
        for window_id in windows_in_sun:
            window_hours[window_id] = window_hours.get(window_id, 0) + 1

    # Sort by hours (most to least)
    sorted_windows = sorted(window_hours.items(), key=lambda x: x[1], reverse=True)

    for window_id, hours in sorted_windows:
        print(f"{window_id:<15} {hours:>2} hours of direct sun")

    if not sorted_windows:
        print("No windows received direct sun during this period")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Test window sun exposure detection locally"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.json",
        help="Path to config file (default: config/default_config.json)",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        help="Sun azimuth in degrees (overrides auto-calculation)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        help="Sun elevation in degrees (overrides auto-calculation)",
    )
    parser.add_argument(
        "--time-range",
        action="store_true",
        help="Test window exposure over time range (6am-8pm)",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Date to test in YYYY-MM-DD format (default: today)",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = Config.from_json_file(args.config)
        print(f"Loaded config from: {args.config}")
        print(f"Windows configured: {len(config.windows)}")
        if config.location:
            print(f"Location: {config.location.latitude:.2f}°N, "
                  f"{config.location.longitude:.2f}°W")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Parse date if provided
    test_date = None
    if args.date:
        try:
            test_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)

    # Run appropriate test
    if args.time_range:
        test_time_range(config, test_date)
    else:
        # Single position test
        if args.azimuth is not None and args.elevation is not None:
            # Use explicit sun position
            azimuth = args.azimuth
            elevation = args.elevation
        else:
            # Auto-calculate from current time and config location
            if config.location is None:
                print("Error: Config must include location data to auto-calculate sun position")
                print("Use --azimuth and --elevation to specify explicit sun position")
                sys.exit(1)

            now = test_date or datetime.now()
            sun_pos = calculate_sun_position(
                config.location.latitude,
                config.location.longitude,
                now,
                timezone_offset=config.location.timezone_offset,
            )
            azimuth = sun_pos.azimuth_deg
            elevation = sun_pos.elevation_deg
            print(f"Using current sun position at {now.strftime('%Y-%m-%d %I:%M %p')}")

        test_single_position(azimuth, elevation, config)


if __name__ == "__main__":
    main()
