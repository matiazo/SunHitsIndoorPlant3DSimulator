"""Geometric utilities for coordinate transforms and vector math.

This module provides functions to convert between different representations
of sun position and direction vectors in the ENU (East-North-Up) coordinate system.

Coordinate System:
    - x = East (positive toward East)
    - y = North (positive toward North)
    - z = Up (positive upward)

Azimuth Convention:
    - 0° = North
    - 90° = East
    - 180° = South
    - 270° = West
    - Clockwise from North
"""

import math

import numpy as np


def sun_direction_from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Convert sun azimuth and elevation to a direction vector.

    Computes the unit vector pointing FROM an observer TOWARD the sun
    in ENU (East-North-Up) coordinates.

    Args:
        azimuth_deg: Sun azimuth in degrees, clockwise from North [0, 360).
            - 0° = North
            - 90° = East
            - 180° = South
            - 270° = West
        elevation_deg: Sun elevation above horizon in degrees [-90, +90].
            - 0° = on horizon
            - 90° = directly overhead (zenith)
            - Negative = below horizon

    Returns:
        Unit vector [x_east, y_north, z_up] pointing toward the sun.

    Examples:
        >>> sun_direction_from_angles(0, 45)  # Sun in North, 45° up
        array([0.        , 0.70710678, 0.70710678])

        >>> sun_direction_from_angles(90, 0)  # Sun in East, on horizon
        array([1., 0., 0.])

        >>> sun_direction_from_angles(180, 30)  # Sun in South, 30° up
        array([ 0.        , -0.8660254 ,  0.5       ])
    """
    az_rad = math.radians(azimuth_deg)
    el_rad = math.radians(elevation_deg)

    cos_el = math.cos(el_rad)
    sin_el = math.sin(el_rad)

    # Convert azimuth (clockwise from North) to ENU components
    # North (y) = cos(azimuth) * cos(elevation)
    # East (x) = sin(azimuth) * cos(elevation)
    # Up (z) = sin(elevation)
    x = cos_el * math.sin(az_rad)  # East component
    y = cos_el * math.cos(az_rad)  # North component
    z = sin_el  # Up component

    return np.array([x, y, z])


def angles_from_sun_direction(direction: np.ndarray) -> tuple[float, float]:
    """Convert a sun direction vector back to azimuth and elevation.

    This is the inverse of sun_direction_from_angles.

    Args:
        direction: Unit vector pointing toward the sun [x, y, z].

    Returns:
        Tuple of (azimuth_deg, elevation_deg).
    """
    x, y, z = direction

    # Elevation is the angle above the horizontal plane
    horizontal_distance = math.sqrt(x * x + y * y)
    elevation_deg = math.degrees(math.atan2(z, horizontal_distance))

    # Azimuth is the angle from North, clockwise
    azimuth_deg = math.degrees(math.atan2(x, y))
    if azimuth_deg < 0:
        azimuth_deg += 360.0

    return azimuth_deg, elevation_deg


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.

    Args:
        v: Input vector.

    Returns:
        Unit vector in the same direction.

    Raises:
        ValueError: If the vector has zero length.
    """
    length = np.linalg.norm(v)
    if length < 1e-10:
        raise ValueError("Cannot normalize zero-length vector")
    return v / length


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Compute dot product of two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Scalar dot product.
    """
    return float(np.dot(a, b))


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cross product of two 3D vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cross product vector a × b.
    """
    return np.cross(a, b)


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the angle between two vectors in degrees.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Angle in degrees [0, 180].
    """
    a_norm = normalize(a)
    b_norm = normalize(b)
    cos_angle = np.clip(dot(a_norm, b_norm), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def project_onto_plane(v: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Project a vector onto a plane defined by its normal.

    Args:
        v: Vector to project.
        plane_normal: Unit normal vector of the plane.

    Returns:
        Component of v that lies in the plane.
    """
    n = normalize(plane_normal)
    return v - dot(v, n) * n


def azimuth_to_direction_2d(azimuth_deg: float) -> np.ndarray:
    """Convert an azimuth angle to a 2D direction vector (xy plane).

    Args:
        azimuth_deg: Azimuth in degrees, clockwise from North.

    Returns:
        2D unit vector [x_east, y_north].
    """
    az_rad = math.radians(azimuth_deg)
    return np.array([math.sin(az_rad), math.cos(az_rad)])


def sun_direction_simplified(
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    wall1_normal_azimuth: float = 210.0,
) -> np.ndarray:
    """Compute sun direction in the simplified coordinate system.

    The simplified coordinate system has:
    - Wall 1 along the X axis at y=0, with outward normal pointing in -Y direction
    - Wall 2 along the Y axis at x=0, with outward normal pointing in -X direction

    This requires rotating the real-world sun direction by the difference between
    the wall's real azimuth and its simplified azimuth (180° for wall 1).

    Args:
        sun_azimuth_deg: Sun azimuth in degrees, clockwise from North (real world).
        sun_elevation_deg: Sun elevation above horizon in degrees.
        wall1_normal_azimuth: Real-world azimuth of wall 1's outward normal.

    Returns:
        Unit vector [x, y, z] pointing toward the sun in simplified coordinates.
    """
    # In simplified coords, wall 1 normal is 180° (due South, -Y)
    # The rotation from ENU to simplified is: wall1_normal_azimuth - 180°
    rotation = wall1_normal_azimuth - 180.0

    # Rotate sun azimuth into simplified frame
    simplified_azimuth = sun_azimuth_deg - rotation

    # Now compute direction using standard formula
    return sun_direction_from_angles(simplified_azimuth, sun_elevation_deg)
