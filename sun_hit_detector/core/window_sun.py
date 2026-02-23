"""Window sun exposure detection module.

This module provides functions to check which windows are receiving direct sunlight
based on sun position. Uses a two-stage approach:
1. Geometric pre-filter: Quick dot product check to filter windows facing away
2. Ray-based validation: Cast rays to verify no obstructions

Main API: check_windows_from_config(sun_azimuth_deg, sun_elevation_deg, config)
"""

import math
from typing import Optional

import numpy as np

from .geometry import dot, sun_direction_simplified
from .models import Config, Window, WindowSunDetail, WindowSunResult
from .ray_casting import ray_window_intersection


def check_window_sun_exposure_geometric(
    window: Window,
    sun_direction: np.ndarray,
) -> tuple[bool, float, float]:
    """Check if window is geometrically facing the sun (fast pre-filter).

    Uses dot product between sun direction and window normal to determine if
    window is facing the sun. This is a fast geometric check that filters out
    windows facing away from the sun before expensive ray tracing.

    Args:
        window: Window object to check.
        sun_direction: Unit vector pointing toward the sun.

    Returns:
        Tuple of (is_facing_sun, angle_deg, intensity_factor) where:
        - is_facing_sun: True if window normal has positive dot product with sun
        - angle_deg: Angle between sun and window normal (0-90 degrees)
        - intensity_factor: Cosine of angle, range [0, 1], represents relative intensity
    """
    # Get window's outward normal vector
    window_normal = window.normal

    # Compute dot product: measures alignment between sun and window normal
    dot_product = dot(sun_direction, window_normal)

    # If dot product <= 0, sun is behind the window (angle >= 90°)
    if dot_product <= 0:
        return False, 90.0, 0.0

    # Calculate angle in degrees
    # dot_product = cos(angle) for unit vectors
    angle_rad = math.acos(min(1.0, dot_product))  # Clamp to avoid numerical issues
    angle_deg = math.degrees(angle_rad)

    # Intensity factor is the dot product itself (cosine of angle)
    # Ranges from 0 (perpendicular) to 1 (directly facing)
    intensity_factor = dot_product

    return True, angle_deg, intensity_factor


def validate_window_sun_with_ray(
    window: Window,
    sun_direction: np.ndarray,
) -> bool:
    """Validate that sun can actually reach window using ray casting.

    Casts a ray from the window center toward the sun to verify that the
    light path is not obstructed. For thick walls, verifies the ray passes
    through the window tunnel correctly.

    Args:
        window: Window object to validate.
        sun_direction: Unit vector pointing toward the sun.

    Returns:
        True if ray path from window to sun is clear, False if obstructed.
    """
    # Start ray from window center (inner face of window)
    ray_origin = window.center

    # Ray direction is toward the sun
    ray_direction = sun_direction

    # Use existing ray_window_intersection to check if ray passes through window
    # This handles thick walls correctly by checking both inner and outer planes
    result = ray_window_intersection(ray_origin, ray_direction, window)

    return result.intersects


def check_windows_from_config(
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    config: Config,
    use_ray_validation: bool = True,
) -> WindowSunResult:
    """Check all windows for sun exposure based on sun position.

    Main API function that combines geometric filtering and optional ray validation
    to determine which windows are receiving direct sunlight.

    Args:
        sun_azimuth_deg: Sun azimuth in degrees (0=North, 90=East, 180=South, 270=West).
        sun_elevation_deg: Sun elevation above horizon in degrees (negative=below horizon).
        config: Configuration object containing windows and coordinate system info.
        use_ray_validation: If True, use ray casting to validate geometric results.

    Returns:
        WindowSunResult with detailed information about each window's sun exposure.
    """
    window_details = {}
    windows_in_sun = []
    reason = None

    # Check if sun is below horizon
    if sun_elevation_deg < 0:
        reason = "sun_below_horizon"
        # Return all windows as not in sun
        for window in config.windows:
            window_details[window.id] = WindowSunDetail(
                window_id=window.id,
                is_in_sun=False,
                sun_angle_to_normal_deg=90.0,
                intensity_factor=0.0,
            )
        return WindowSunResult(
            windows_in_sun=[],
            window_details=window_details,
            sun_azimuth_deg=sun_azimuth_deg,
            sun_elevation_deg=sun_elevation_deg,
            sun_direction=None,
            reason=reason,
        )

    # Get wall 1 normal azimuth for coordinate system transformation
    wall1_normal = config.walls[0].outward_normal_azimuth_deg if config.walls else 210.0

    # Calculate sun direction in simplified coordinate system
    sun_direction = sun_direction_simplified(
        sun_azimuth_deg,
        sun_elevation_deg,
        wall1_normal,
    )

    # Check each window
    for window in config.windows:
        # Stage 1: Geometric pre-filter
        is_facing, angle_deg, intensity = check_window_sun_exposure_geometric(
            window, sun_direction
        )

        # Stage 2: Ray validation (optional)
        is_in_sun = False
        if is_facing:
            if use_ray_validation:
                # Use ray casting to verify sun can actually reach window
                is_in_sun = validate_window_sun_with_ray(window, sun_direction)
            else:
                # Skip ray validation, use geometric result only
                is_in_sun = True

        # Store results
        window_details[window.id] = WindowSunDetail(
            window_id=window.id,
            is_in_sun=is_in_sun,
            sun_angle_to_normal_deg=angle_deg,
            intensity_factor=intensity,
        )

        # Add to windows_in_sun list
        if is_in_sun:
            windows_in_sun.append(window.id)

    # Set reason if no windows in sun
    if not windows_in_sun and sun_elevation_deg >= 0:
        reason = "no_windows_facing_sun"

    return WindowSunResult(
        windows_in_sun=windows_in_sun,
        window_details=window_details,
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        sun_direction=sun_direction,
        reason=reason,
    )
