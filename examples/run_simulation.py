#!/usr/bin/env python3
"""Example script demonstrating the sun-plant simulator.

This script shows how to:
1. Load configuration and sun data
2. Run a single hit test
3. Run a time-range simulation
4. Generate 3D visualizations

Usage:
    python examples/run_simulation.py
"""

import json
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sun_plant_simulator.core.hit_test import check_sun_hits_plant, get_detailed_hit_info
from sun_plant_simulator.core.models import Config
from sun_plant_simulator.simulator.time_range import (
    load_sun_data_from_json,
    simulate_time_range,
)
from sun_plant_simulator.visualization.scene_builder import build_scene, visualize_hit_test


def main():
    """Run example simulation."""
    # Paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "default_config.json"
    sun_data_path = project_root / "data" / "sample_sun_data.json"

    print("=" * 60)
    print("Sun Hits Indoor Plant 3D Simulator - Example")
    print("=" * 60)

    # Load configuration
    print("\n1. Loading configuration...")
    config = Config.from_json_file(config_path)
    print(f"   - Loaded {len(config.windows)} windows")
    print(f"   - Plant at ({config.plant.center_x}, {config.plant.center_y})")
    print(f"   - Plant height: {config.plant.z_min} to {config.plant.z_max} m")

    # Single hit test
    print("\n2. Running single hit test...")
    test_azimuth = 210  # SW direction (aligned with wall 1)
    test_elevation = 30

    result = check_sun_hits_plant(
        sun_azimuth_deg=test_azimuth,
        sun_elevation_deg=test_elevation,
        plant=config.plant,
        windows=config.windows,
        n_angular=config.simulation.sample_points_angular,
        n_vertical=config.simulation.sample_points_vertical,
    )

    print(f"   - Sun position: azimuth={test_azimuth}°, elevation={test_elevation}°")
    print(f"   - Result: {'HIT' if result.is_hit else 'MISS'}")
    if result.is_hit:
        print(f"   - Window: {result.window_id}")
        print(f"   - Hit points: {len(result.hit_points)}")
    else:
        print(f"   - Reason: {result.reason}")

    # Time-range simulation
    print("\n3. Running time-range simulation...")
    if sun_data_path.exists():
        sun_data = load_sun_data_from_json(sun_data_path)
        print(f"   - Loaded {len(sun_data)} sun data points")

        sim_result = simulate_time_range(sun_data, config)

        print(f"   - Total timestamps: {sim_result.total_timestamps}")
        print(f"   - Hit count: {sim_result.hit_count}")
        print(f"   - Miss count: {sim_result.miss_count}")
        print(f"   - Hit percentage: {sim_result.hit_percentage:.1f}%")
        print(f"   - Hit intervals: {len(sim_result.hit_intervals)}")

        for interval in sim_result.hit_intervals:
            print(f"     - {interval.start_timestamp} to {interval.end_timestamp}")
            print(f"       (window: {interval.window_id}, {interval.n_timestamps} timestamps)")
    else:
        print(f"   - Warning: Sun data file not found at {sun_data_path}")

    # Visualization
    print("\n4. Creating visualization...")
    try:
        fig = visualize_hit_test(config, test_azimuth, test_elevation)

        # Save to HTML file
        output_path = project_root / "examples" / "visualization.html"
        fig.write_html(str(output_path))
        print(f"   - Saved interactive visualization to: {output_path}")
        print("   - Open this file in a web browser to explore the 3D scene")
    except Exception as e:
        print(f"   - Visualization skipped (error: {e})")
        print("   - Make sure plotly is installed: pip install plotly")

    # Detailed hit info (for debugging)
    print("\n5. Detailed hit analysis...")
    details = get_detailed_hit_info(
        test_azimuth, test_elevation, config.plant, config.windows
    )
    print(f"   - Sample points tested: {details['n_sample_points']}")
    print(f"   - Points receiving light: {details['n_hit_points']}")
    print(f"   - Sun direction: [{details['sun_direction'][0]:.3f}, "
          f"{details['sun_direction'][1]:.3f}, {details['sun_direction'][2]:.3f}]")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
