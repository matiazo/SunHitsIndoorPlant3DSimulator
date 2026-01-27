"""Ray casting utilities for window intersection detection.

This module provides functions to determine if a ray (representing sunlight
traveling from the sun toward a point on the plant) passes through a window.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .models import Window


@dataclass
class RayIntersection:
    """Result of a ray-window intersection test.

    Attributes:
        intersects: Whether the ray intersects the window.
        point: The intersection point (if intersects is True).
        t: The parameter t such that intersection = origin + t * direction.
        local_h: Horizontal position on window [-width/2, width/2].
        local_v: Vertical position on window [-height/2, height/2].
    """

    intersects: bool
    point: Optional[np.ndarray] = None
    t: Optional[float] = None
    local_h: Optional[float] = None
    local_v: Optional[float] = None


def ray_intersects_window(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    window: Window,
    epsilon: float = 1e-10,
) -> bool:
    """Check if a ray passes through a window rectangle.

    The ray represents sunlight traveling from the sun toward a point on the plant.
    For a hit to occur:
    1. The ray must intersect the plane of the window
    2. The intersection must be within the window's rectangular bounds
    3. The ray must be going outward (toward the sun, not into the room)

    Args:
        ray_origin: Starting point of the ray (point on plant).
        ray_direction: Direction of the ray (toward the sun), should be unit vector.
        window: The window to test against.
        epsilon: Small value for numerical comparisons.

    Returns:
        True if the ray passes through the window, False otherwise.
    """
    result = ray_window_intersection(ray_origin, ray_direction, window, epsilon)
    return result.intersects


def ray_window_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    window: Window,
    epsilon: float = 1e-10,
) -> RayIntersection:
    """Compute detailed ray-window intersection.

    Uses the ray-plane intersection formula:
        t = (plane_point - ray_origin) · plane_normal / (ray_direction · plane_normal)
        intersection = ray_origin + t * ray_direction

    Then checks if the intersection point is within the window bounds.

    Args:
        ray_origin: Starting point of the ray.
        ray_direction: Direction of the ray (should be normalized).
        window: The window to test against.
        epsilon: Small value for numerical comparisons.

    Returns:
        RayIntersection with details about the intersection.
    """
    normal = window.normal

    # Compute denominator: ray_direction · plane_normal
    denom = np.dot(ray_direction, normal)

    # Check if ray is parallel to window plane
    if abs(denom) < epsilon:
        return RayIntersection(intersects=False)

    # Check if ray is going outward (same direction as normal)
    # The normal points outward from the room, and we want the ray to be
    # going toward the outside (toward the sun)
    if denom <= 0:
        # Ray is pointing into the room or parallel, not toward outside
        return RayIntersection(intersects=False)

    # Compute t: distance along ray to plane intersection
    # Plane equation: (point - center) · normal = 0
    # Ray equation: point = origin + t * direction
    # Substituting: (origin + t * direction - center) · normal = 0
    # t * (direction · normal) = (center - origin) · normal
    # t = (center - origin) · normal / (direction · normal)
    t = np.dot(window.center - ray_origin, normal) / denom

    # Check if intersection is in front of ray origin (t > 0)
    if t < 0:
        return RayIntersection(intersects=False)

    # Compute intersection point
    intersection = ray_origin + t * ray_direction

    # Project intersection onto window's local coordinate system
    # to check if it's within bounds
    offset = intersection - window.center

    # Horizontal position (along window width)
    local_h = np.dot(offset, window.horizontal_axis)

    # Vertical position (along window height)
    local_v = offset[2]  # z-component relative to center

    # Check if within window bounds
    within_h = abs(local_h) <= window.width / 2 + epsilon
    within_v = abs(local_v) <= window.height / 2 + epsilon

    if within_h and within_v:
        return RayIntersection(
            intersects=True,
            point=intersection,
            t=t,
            local_h=local_h,
            local_v=local_v,
        )
    else:
        return RayIntersection(
            intersects=False,
            point=intersection,
            t=t,
            local_h=local_h,
            local_v=local_v,
        )


def ray_hits_any_window(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    windows: list[Window],
) -> tuple[bool, Optional[str]]:
    """Check if a ray passes through any of the provided windows.

    Args:
        ray_origin: Starting point of the ray.
        ray_direction: Direction of the ray.
        windows: List of windows to test.

    Returns:
        Tuple of (hit_any, window_id) where window_id is the ID of the first
        window hit (or None if no hit).
    """
    for window in windows:
        if ray_intersects_window(ray_origin, ray_direction, window):
            return True, window.id
    return False, None
