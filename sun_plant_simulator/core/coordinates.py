"""Coordinate conversion helpers.

This module provides utilities to convert between different ways of
specifying positions in a room:
- Distance from walls
- x/y (ENU) coordinates
"""

import math
from typing import Optional

import numpy as np


def wall_normal_to_inward_direction(wall_normal_azimuth_deg: float) -> np.ndarray:
    """Get the inward direction (into the room) from a wall's outward normal.

    Args:
        wall_normal_azimuth_deg: Outward normal azimuth in degrees.

    Returns:
        2D unit vector pointing into the room (opposite of outward normal).
    """
    # Outward normal direction
    az_rad = math.radians(wall_normal_azimuth_deg)
    outward = np.array([math.sin(az_rad), math.cos(az_rad)])

    # Inward is opposite
    return -outward


def position_from_wall_distances(
    dist_from_wall1: float,
    dist_from_wall2: float,
    wall1_normal_azimuth: float,
    wall2_normal_azimuth: float,
    corner_x: float = 0.0,
    corner_y: float = 0.0,
) -> tuple[float, float]:
    """Convert distances from two walls to x/y coordinates.

    The corner is where the two walls meet. Distances are measured
    perpendicular to each wall, going into the room.

    Args:
        dist_from_wall1: Distance from wall 1 in meters.
        dist_from_wall2: Distance from wall 2 in meters.
        wall1_normal_azimuth: Wall 1 outward normal azimuth (degrees).
        wall2_normal_azimuth: Wall 2 outward normal azimuth (degrees).
        corner_x: X coordinate of the corner where walls meet.
        corner_y: Y coordinate of the corner where walls meet.

    Returns:
        Tuple of (x, y) coordinates in meters.

    Example:
        >>> # Walls at 210° and 307° normals, corner at origin
        >>> x, y = position_from_wall_distances(
        ...     dist_from_wall1=1.5,
        ...     dist_from_wall2=2.0,
        ...     wall1_normal_azimuth=210,
        ...     wall2_normal_azimuth=307,
        ... )
    """
    # Get inward directions for each wall
    dir1 = wall_normal_to_inward_direction(wall1_normal_azimuth)
    dir2 = wall_normal_to_inward_direction(wall2_normal_azimuth)

    # Position = corner + dist1 * dir1 + dist2 * dir2
    corner = np.array([corner_x, corner_y])
    position = corner + dist_from_wall1 * dir1 + dist_from_wall2 * dir2

    return float(position[0]), float(position[1])


def wall_distances_from_position(
    x: float,
    y: float,
    wall1_normal_azimuth: float,
    wall2_normal_azimuth: float,
    corner_x: float = 0.0,
    corner_y: float = 0.0,
) -> tuple[float, float]:
    """Convert x/y coordinates to distances from two walls.

    Inverse of position_from_wall_distances.

    Args:
        x: X coordinate in meters.
        y: Y coordinate in meters.
        wall1_normal_azimuth: Wall 1 outward normal azimuth (degrees).
        wall2_normal_azimuth: Wall 2 outward normal azimuth (degrees).
        corner_x: X coordinate of the corner where walls meet.
        corner_y: Y coordinate of the corner where walls meet.

    Returns:
        Tuple of (dist_from_wall1, dist_from_wall2) in meters.
    """
    # Get inward directions
    dir1 = wall_normal_to_inward_direction(wall1_normal_azimuth)
    dir2 = wall_normal_to_inward_direction(wall2_normal_azimuth)

    # Offset from corner
    offset = np.array([x - corner_x, y - corner_y])

    # Solve: offset = d1 * dir1 + d2 * dir2
    # This is a 2x2 linear system: [dir1 | dir2] * [d1, d2]^T = offset
    A = np.column_stack([dir1, dir2])
    distances = np.linalg.solve(A, offset)

    return float(distances[0]), float(distances[1])


def print_coordinate_info(
    wall1_normal_azimuth: float = 210,
    wall2_normal_azimuth: float = 307,
) -> None:
    """Print helpful information about the coordinate system.

    Args:
        wall1_normal_azimuth: Wall 1 outward normal azimuth.
        wall2_normal_azimuth: Wall 2 outward normal azimuth.
    """
    dir1 = wall_normal_to_inward_direction(wall1_normal_azimuth)
    dir2 = wall_normal_to_inward_direction(wall2_normal_azimuth)

    print("Coordinate System Information")
    print("=" * 40)
    print(f"\nWall 1: outward normal at {wall1_normal_azimuth}°")
    print(f"  - Inward direction: ({dir1[0]:.3f}, {dir1[1]:.3f})")
    print(f"  - Wall faces: {(wall1_normal_azimuth + 180) % 360:.0f}° (into room)")

    print(f"\nWall 2: outward normal at {wall2_normal_azimuth}°")
    print(f"  - Inward direction: ({dir2[0]:.3f}, {dir2[1]:.3f})")
    print(f"  - Wall faces: {(wall2_normal_azimuth + 180) % 360:.0f}° (into room)")

    # Angle between walls
    angle = math.degrees(math.acos(np.clip(np.dot(dir1, dir2), -1, 1)))
    print(f"\nAngle between walls: {angle:.1f}°")

    print("\nTo convert wall distances to x/y:")
    print("  from sun_plant_simulator.core.coordinates import position_from_wall_distances")
    print(f"  x, y = position_from_wall_distances(dist1, dist2, {wall1_normal_azimuth}, {wall2_normal_azimuth})")


# Convenience function with your specific wall angles
def plant_position_from_wall_distances(
    dist_from_wall1: float,
    dist_from_wall2: float,
    corner_x: float = 0.0,
    corner_y: float = 0.0,
) -> tuple[float, float]:
    """Convert plant distances from walls to x/y using default wall angles (210°, 307°).

    This is a convenience function with your room's wall angles pre-configured.

    Args:
        dist_from_wall1: Distance from wall 1 (210° normal) in meters.
        dist_from_wall2: Distance from wall 2 (307° normal) in meters.
        corner_x: X coordinate of corner (default 0).
        corner_y: Y coordinate of corner (default 0).

    Returns:
        Tuple of (x, y) coordinates.

    Example:
        >>> x, y = plant_position_from_wall_distances(1.5, 2.0)
        >>> print(f"Plant at x={x:.2f}m, y={y:.2f}m")
    """
    return position_from_wall_distances(
        dist_from_wall1=dist_from_wall1,
        dist_from_wall2=dist_from_wall2,
        wall1_normal_azimuth=210,
        wall2_normal_azimuth=307,
        corner_x=corner_x,
        corner_y=corner_y,
    )
