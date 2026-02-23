"""Main hit-test algorithm for sun-plant simulation.

This module contains the core algorithm that determines whether direct sunlight
passes through a window and strikes the plant.
"""

import math
from typing import Optional

import numpy as np

from .geometry import sun_direction_from_angles, sun_direction_simplified
from .models import Config, HitResult, Plant, Window
from .ray_casting import ray_intersects_window, ray_window_intersection


def generate_plant_sample_points(
    plant: Plant,
    n_angular: int = 8,
    n_vertical: int = 3,
) -> list[np.ndarray]:
    """Generate sample points on the plant cylinder surface.

    Creates a grid of test points on the surface of the cylindrical plant model.
    Points are distributed around the circumference and along the height.
    Additional points are added at the center of the top surface.

    Args:
        plant: Plant geometry definition.
        n_angular: Number of angular divisions around the cylinder (default 8).
        n_vertical: Number of vertical divisions along the cylinder (default 3).

    Returns:
        List of 3D points on the plant surface.
    """
    points = []

    # Points around the cylinder surface
    for i in range(n_angular):
        angle = 2 * math.pi * i / n_angular
        x = plant.center_x + plant.radius * math.cos(angle)
        y = plant.center_y + plant.radius * math.sin(angle)

        for j in range(n_vertical):
            if n_vertical > 1:
                z = plant.z_min + (plant.z_max - plant.z_min) * j / (n_vertical - 1)
            else:
                z = (plant.z_min + plant.z_max) / 2
            points.append(np.array([x, y, z]))

    # Center point at the top of the plant
    points.append(np.array([plant.center_x, plant.center_y, plant.z_max]))

    # Center point at the middle height
    points.append(
        np.array([plant.center_x, plant.center_y, (plant.z_min + plant.z_max) / 2])
    )

    return points


def check_sun_hits_plant(
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    plant: Plant,
    windows: list[Window],
    n_angular: int = 8,
    n_vertical: int = 3,
    wall1_normal_azimuth: float = 210.0,
) -> HitResult:
    """Determine if direct sunlight hits the plant through any window.

    This is the main algorithm. For each sample point on the plant cylinder,
    it casts a ray toward the sun and checks if that ray passes through any window.

    The algorithm uses a simplified coordinate system where walls are axis-aligned.
    The sun direction is rotated from real-world coordinates into this simplified frame.

    Args:
        sun_azimuth_deg: Sun azimuth in degrees, clockwise from North [0, 360).
        sun_elevation_deg: Sun elevation above horizon in degrees [-90, +90].
        plant: Plant geometry definition.
        windows: List of window definitions.
        n_angular: Number of angular divisions for plant sampling.
        n_vertical: Number of vertical divisions for plant sampling.
        wall1_normal_azimuth: Real-world azimuth of wall 1's outward normal.

    Returns:
        HitResult containing:
        - is_hit: True if any sample point receives direct sunlight
        - window_id: ID of the window through which light passes (if hit)
        - hit_points: List of sample points that receive sunlight
        - sun_direction: Direction vector toward the sun
        - reason: Explanation if not hit (e.g., "sun_below_horizon")
    """
    # Sun below horizon = no direct sunlight possible
    if sun_elevation_deg <= 0:
        return HitResult(is_hit=False, reason="sun_below_horizon")

    # Compute sun direction vector in simplified coordinate system
    sun_dir = sun_direction_simplified(sun_azimuth_deg, sun_elevation_deg, wall1_normal_azimuth)

    # Generate sample points on the plant
    sample_points = generate_plant_sample_points(plant, n_angular, n_vertical)

    hit_points = []
    hit_window_id: Optional[str] = None

    # Test each sample point
    for point in sample_points:
        for window in windows:
            if ray_intersects_window(point, sun_dir, window):
                hit_points.append(point)
                if hit_window_id is None:
                    hit_window_id = window.id
                break  # Point can only be hit through one window

    if hit_points:
        return HitResult(
            is_hit=True,
            window_id=hit_window_id,
            hit_points=hit_points,
            sun_direction=sun_dir,
        )
    else:
        return HitResult(
            is_hit=False,
            sun_direction=sun_dir,
            reason="no_window_path",
        )


def check_sun_hits_plant_from_config(
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    config: Config,
) -> HitResult:
    """Convenience function to run hit test using a Config object.

    Args:
        sun_azimuth_deg: Sun azimuth in degrees.
        sun_elevation_deg: Sun elevation in degrees.
        config: Configuration containing plant, windows, and simulation params.

    Returns:
        HitResult with hit information.
    """
    return check_sun_hits_plant(
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        plant=config.plant,
        windows=config.windows,
        n_angular=config.simulation.sample_points_angular,
        n_vertical=config.simulation.sample_points_vertical,
    )


def get_detailed_hit_info(
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    plant: Plant,
    windows: list[Window],
    n_angular: int = 8,
    n_vertical: int = 3,
) -> dict:
    """Get detailed information about sun-plant-window geometry.

    Useful for debugging and visualization. Returns information about
    each sample point and its relationship to each window.

    Args:
        sun_azimuth_deg: Sun azimuth in degrees.
        sun_elevation_deg: Sun elevation in degrees.
        plant: Plant geometry.
        windows: List of windows.
        n_angular: Angular sampling.
        n_vertical: Vertical sampling.

    Returns:
        Dictionary with detailed hit information including per-point analysis.
    """
    if sun_elevation_deg <= 0:
        return {
            "is_hit": False,
            "reason": "sun_below_horizon",
            "sun_azimuth_deg": sun_azimuth_deg,
            "sun_elevation_deg": sun_elevation_deg,
            "points": [],
        }

    sun_dir = sun_direction_from_angles(sun_azimuth_deg, sun_elevation_deg)
    sample_points = generate_plant_sample_points(plant, n_angular, n_vertical)

    point_details = []
    any_hit = False
    first_hit_window = None

    for i, point in enumerate(sample_points):
        point_info = {
            "index": i,
            "position": point.tolist(),
            "windows": [],
        }

        for window in windows:
            result = ray_window_intersection(point, sun_dir, window)
            window_info = {
                "window_id": window.id,
                "intersects": result.intersects,
            }
            if result.point is not None:
                window_info["intersection_point"] = result.point.tolist()
                window_info["t"] = result.t
                window_info["local_h"] = result.local_h
                window_info["local_v"] = result.local_v

            point_info["windows"].append(window_info)

            if result.intersects:
                any_hit = True
                if first_hit_window is None:
                    first_hit_window = window.id

        point_details.append(point_info)

    return {
        "is_hit": any_hit,
        "window_id": first_hit_window,
        "sun_azimuth_deg": sun_azimuth_deg,
        "sun_elevation_deg": sun_elevation_deg,
        "sun_direction": sun_dir.tolist(),
        "n_sample_points": len(sample_points),
        "n_hit_points": sum(
            1
            for p in point_details
            if any(w["intersects"] for w in p["windows"])
        ),
        "points": point_details,
    }
