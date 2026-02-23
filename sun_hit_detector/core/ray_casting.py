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


def _intersect_axis_aligned_plane(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    plane_axis: int,
    plane_coord: float,
    window: Window,
    epsilon: float = 1e-10,
) -> RayIntersection:
    """Intersect ray with an axis-aligned plane and check window bounds.

    Args:
        ray_origin: Starting point of the ray.
        ray_direction: Direction of the ray (should be normalized).
        plane_axis: Which axis the plane is perpendicular to (0=x, 1=y).
        plane_coord: Coordinate of the plane on the given axis.
        window: The window to check bounds against.
        epsilon: Small value for numerical comparisons.

    Returns:
        RayIntersection with details about the intersection.
    """
    # Check if ray is parallel to plane
    if abs(ray_direction[plane_axis]) < epsilon:
        return RayIntersection(intersects=False)

    # Compute t: distance along ray to plane intersection
    t = (plane_coord - ray_origin[plane_axis]) / ray_direction[plane_axis]

    # Check if intersection is in front of ray origin (t > 0)
    if t < 0:
        return RayIntersection(intersects=False)

    # Compute intersection point
    intersection = ray_origin + t * ray_direction

    # Check if within window bounds
    # For wall 1 (plane_axis=1, y=const): check x and z
    # For wall 2 (plane_axis=0, x=const): check y and z
    if plane_axis == 1:  # Wall 1
        local_h = intersection[0] - window.center[0]  # x offset
    else:  # Wall 2
        local_h = intersection[1] - window.center[1]  # y offset

    local_v = intersection[2] - window.center[2]  # z offset

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


def ray_window_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    window: Window,
    epsilon: float = 1e-10,
) -> RayIntersection:
    """Compute detailed ray-window intersection.

    For windows with wall_thickness > 0, models the window as a rectangular tunnel
    (box) through the wall. The ray must pass through BOTH the inner and outer
    window planes within bounds to count as a hit.

    For windows with wall_thickness == 0, uses the simple thin-plane model.

    The window positions use a simplified coordinate system where:
    - Windows on wall 1 are at y=0 (wall 1 plane is y=window.center[1])
    - Windows on wall 2 are at x=0 (wall 2 plane is x=window.center[0])

    The wall normal azimuth determines if the sun can shine through
    (sun must be on the outside of the wall).

    Args:
        ray_origin: Starting point of the ray.
        ray_direction: Direction of the ray (should be normalized).
        window: The window to test against.
        epsilon: Small value for numerical comparisons.

    Returns:
        RayIntersection with details about the intersection.
    """
    wall_normal = window.normal  # Direction wall faces (for sun angle check)

    # Check if sun is on the correct side of the wall (outside the room)
    # The sun direction points toward the sun; dot with outward normal should be > 0
    sun_side_check = np.dot(ray_direction, wall_normal)
    if sun_side_check <= 0:
        # Sun is on the inside of the wall or parallel - can't shine through
        return RayIntersection(intersects=False)

    # Determine plane properties based on window axis (simplified axis-aligned geometry)
    if window.axis == "x":
        plane_axis = 1
    elif window.axis == "y":
        plane_axis = 0
    else:
        is_wall1 = abs(window.center[1]) < abs(window.center[0])
        plane_axis = 1 if is_wall1 else 0

    # Inner plane coordinate (where window.center is located)
    inner_plane_coord = window.center[plane_axis]

    # If wall has no thickness, use simple single-plane intersection
    if window.wall_thickness <= 0:
        return _intersect_axis_aligned_plane(
            ray_origin, ray_direction, plane_axis, inner_plane_coord, window, epsilon
        )

    # Wall has thickness - model as tunnel (two planes)
    # The outward normal determines which direction is "outside"
    # For wall 1 (y=0): normal points outward (-Y direction in simplified coords)
    # For wall 2 (x=0): normal points outward (-X direction in simplified coords)
    # Outer plane is offset in the outward normal direction

    # In the simplified coordinate system:
    # - Wall 1: inner at y=0, outer at y=-thickness (outside is -Y)
    # - Wall 2: inner at x=0, outer at x=-thickness (outside is -X)
    outer_plane_coord = inner_plane_coord - window.wall_thickness

    # Test inner plane intersection
    inner_result = _intersect_axis_aligned_plane(
        ray_origin, ray_direction, plane_axis, inner_plane_coord, window, epsilon
    )

    # Test outer plane intersection
    outer_result = _intersect_axis_aligned_plane(
        ray_origin, ray_direction, plane_axis, outer_plane_coord, window, epsilon
    )

    # Both must intersect within window bounds for light to pass through tunnel
    if inner_result.intersects and outer_result.intersects:
        # Return the inner intersection (where light enters room)
        return inner_result
    else:
        # At least one intersection failed - light blocked by wall thickness
        return RayIntersection(intersects=False)


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
