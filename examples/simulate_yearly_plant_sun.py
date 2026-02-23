#!/usr/bin/env python3
"""Year-long simulation to check if side windows (wall_2) illuminate the plant.

This script simulates sun exposure for every day of the year to determine if
the plant ever receives direct sunlight through the side windows (wall_2 at
azimuth 300°).

Usage:
    # Run full year simulation
    python simulate_yearly_plant_sun.py

    # Run with custom config
    python simulate_yearly_plant_sun.py --config /path/to/config.json

    # Test specific year
    python simulate_yearly_plant_sun.py --year 2026

    # Higher time resolution (every 15 minutes instead of hourly)
    python simulate_yearly_plant_sun.py --interval 15
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sun_hit_detector.core.models import Config
from sun_hit_detector.core.hit_test import check_sun_hits_plant
from sun_hit_detector.core.sun_position import calculate_sun_position


def simulate_year(
    config: Config,
    year: int = 2026,
    interval_minutes: int = 60,
    target_wall_azimuth: float = 300.0,
):
    """Simulate plant sun exposure for entire year.

    Args:
        config: Configuration object with plant and window definitions
        year: Year to simulate
        interval_minutes: Time interval in minutes (60 = hourly, 15 = every 15 min)
        target_wall_azimuth: Azimuth of wall to focus on (300 = wall_2)

    Returns:
        Dictionary with simulation results
    """
    if config.location is None:
        raise ValueError("Config must include location data for year simulation")

    # Identify windows on target wall
    target_windows = [
        w for w in config.windows
        if abs(w.wall_normal_azimuth - target_wall_azimuth) < 10
    ]

    if not target_windows:
        print(f"Warning: No windows found near azimuth {target_wall_azimuth}")
        target_windows = config.windows

    target_window_ids = {w.id for w in target_windows}

    print(f"Target wall azimuth: {target_wall_azimuth}°")
    print(f"Target windows: {', '.join(target_window_ids)}")
    print(f"Simulating year {year} at {interval_minutes}-minute intervals...")
    print()

    # Results tracking
    days_with_sun = []
    hits_by_window = defaultdict(list)  # window_id -> list of (date, time) tuples
    hits_by_month = defaultdict(int)  # month -> count of days with sun
    total_tests = 0
    hits_from_target = 0
    hits_from_other = 0

    # Simulate each day
    start_date = datetime(year, 1, 1)
    for day_offset in range(365):
        current_date = start_date + timedelta(days=day_offset)
        day_had_sun_from_target = False
        day_hits = []

        # Test every interval throughout the day (6am to 8pm)
        for hour in range(6, 21):
            for minute in range(0, 60, interval_minutes):
                test_time = current_date.replace(
                    hour=hour,
                    minute=minute,
                    second=0,
                    microsecond=0
                )

                # Calculate sun position
                sun_pos = calculate_sun_position(
                    config.location.latitude,
                    config.location.longitude,
                    test_time,
                    timezone_offset=config.location.timezone_offset,
                )

                # Skip if sun below horizon
                if sun_pos.elevation_deg < 0:
                    total_tests += 1
                    continue

                # Check if plant gets sun
                result = check_sun_hits_plant(
                    sun_azimuth_deg=sun_pos.azimuth_deg,
                    sun_elevation_deg=sun_pos.elevation_deg,
                    plant=config.plant,
                    windows=config.windows,
                    n_angular=config.simulation.sample_points_angular,
                    n_vertical=config.simulation.sample_points_vertical,
                )

                total_tests += 1

                if result.is_hit and result.window_id:
                    if result.window_id in target_window_ids:
                        # Hit from target wall!
                        hits_from_target += 1
                        day_had_sun_from_target = True
                        time_str = test_time.strftime("%I:%M %p")
                        day_hits.append({
                            'time': time_str,
                            'window': result.window_id,
                            'azimuth': sun_pos.azimuth_deg,
                            'elevation': sun_pos.elevation_deg,
                        })
                        hits_by_window[result.window_id].append(
                            (current_date, time_str)
                        )
                    else:
                        hits_from_other += 1

        # Record day if it had sun from target windows
        if day_had_sun_from_target:
            days_with_sun.append({
                'date': current_date,
                'hits': day_hits,
            })
            hits_by_month[current_date.month] += 1

        # Progress indicator
        if (day_offset + 1) % 30 == 0:
            progress = ((day_offset + 1) / 365) * 100
            print(f"Progress: {progress:.0f}% ({day_offset + 1}/365 days)")

    return {
        'target_wall_azimuth': target_wall_azimuth,
        'target_window_ids': list(target_window_ids),
        'year': year,
        'interval_minutes': interval_minutes,
        'total_tests': total_tests,
        'hits_from_target': hits_from_target,
        'hits_from_other': hits_from_other,
        'days_with_sun': days_with_sun,
        'hits_by_window': dict(hits_by_window),
        'hits_by_month': dict(hits_by_month),
    }


def print_results(results: dict):
    """Print formatted simulation results.

    Args:
        results: Dictionary from simulate_year()
    """
    print("\n" + "=" * 100)
    print("YEAR-LONG PLANT SUN EXPOSURE SIMULATION")
    print("=" * 100)

    print(f"\nTarget Wall: Azimuth {results['target_wall_azimuth']}°")
    print(f"Target Windows: {', '.join(results['target_window_ids'])}")
    print(f"Year: {results['year']}")
    print(f"Time Interval: {results['interval_minutes']} minutes")

    print("\n" + "-" * 100)
    print("SUMMARY")
    print("-" * 100)

    total_tests = results['total_tests']
    hits_target = results['hits_from_target']
    hits_other = results['hits_from_other']
    days_with_sun = len(results['days_with_sun'])

    print(f"Total time points tested: {total_tests:,}")
    print(f"Plant illuminated from target windows: {hits_target:,} times ({hits_target/total_tests*100:.2f}%)")
    print(f"Plant illuminated from other windows: {hits_other:,} times ({hits_other/total_tests*100:.2f}%)")
    print(f"Days with sun from target windows: {days_with_sun} out of 365 days")

    if days_with_sun == 0:
        print("\n" + "!" * 100)
        print("RESULT: The plant NEVER receives direct sunlight from the side windows (wall_2) throughout the year!")
        print("!" * 100)
        return

    print("\n" + "-" * 100)
    print("DAYS WITH SUN FROM TARGET WINDOWS")
    print("-" * 100)

    # Show first 10 days as examples
    print("\nFirst 10 days with sun exposure:")
    for i, day_info in enumerate(results['days_with_sun'][:10]):
        date = day_info['date']
        hits = day_info['hits']
        print(f"\n{i+1}. {date.strftime('%A, %B %d, %Y')}")
        print(f"   Times: {', '.join([h['time'] for h in hits[:5]])}", end="")
        if len(hits) > 5:
            print(f" ... (+{len(hits)-5} more)")
        else:
            print()
        print(f"   Windows: {', '.join(set([h['window'] for h in hits]))}")

    # Monthly breakdown
    print("\n" + "-" * 100)
    print("MONTHLY BREAKDOWN")
    print("-" * 100)
    print(f"{'Month':<15} {'Days with Sun':<15} {'% of Month'}")
    print("-" * 100)

    month_names = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for month in range(1, 13):
        days = results['hits_by_month'].get(month, 0)
        total_days = days_in_month[month - 1]
        pct = (days / total_days) * 100 if total_days > 0 else 0
        print(f"{month_names[month-1]:<15} {days:<15} {pct:.1f}%")

    # Per-window breakdown
    print("\n" + "-" * 100)
    print("PER-WINDOW BREAKDOWN")
    print("-" * 100)

    hits_by_window = results['hits_by_window']
    if hits_by_window:
        for window_id in sorted(hits_by_window.keys()):
            hits = hits_by_window[window_id]
            unique_days = len(set(date for date, _ in hits))
            print(f"{window_id}: {len(hits)} hits across {unique_days} days")
    else:
        print("No hits from target windows")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate plant sun exposure for entire year"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.json",
        help="Path to config file",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Year to simulate (default: 2026)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Time interval in minutes (default: 60 = hourly)",
    )
    parser.add_argument(
        "--wall-azimuth",
        type=float,
        default=300.0,
        help="Target wall azimuth to focus on (default: 300 = wall_2)",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = Config.from_json_file(args.config)
        print(f"Loaded config from: {args.config}")
        print(f"Location: {config.location.latitude:.4f}°N, {config.location.longitude:.4f}°W")
        print(f"Total windows: {len(config.windows)}")
        print()
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Run simulation
    try:
        results = simulate_year(
            config=config,
            year=args.year,
            interval_minutes=args.interval,
            target_wall_azimuth=args.wall_azimuth,
        )

        # Print results
        print_results(results)

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
