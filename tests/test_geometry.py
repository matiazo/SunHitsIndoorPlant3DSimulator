"""Tests for the geometry module."""

import math

import numpy as np
import pytest

from sun_plant_simulator.core.geometry import (
    angle_between_vectors,
    angles_from_sun_direction,
    normalize,
    sun_direction_from_angles,
)


class TestSunDirectionFromAngles:
    """Tests for sun_direction_from_angles function."""

    def test_sun_north_horizon(self):
        """Sun in North (az=0°), on horizon (el=0°)."""
        direction = sun_direction_from_angles(0, 0)
        expected = np.array([0.0, 1.0, 0.0])  # Pointing North
        np.testing.assert_array_almost_equal(direction, expected)

    def test_sun_east_horizon(self):
        """Sun in East (az=90°), on horizon (el=0°)."""
        direction = sun_direction_from_angles(90, 0)
        expected = np.array([1.0, 0.0, 0.0])  # Pointing East
        np.testing.assert_array_almost_equal(direction, expected)

    def test_sun_south_horizon(self):
        """Sun in South (az=180°), on horizon (el=0°)."""
        direction = sun_direction_from_angles(180, 0)
        expected = np.array([0.0, -1.0, 0.0])  # Pointing South
        np.testing.assert_array_almost_equal(direction, expected)

    def test_sun_west_horizon(self):
        """Sun in West (az=270°), on horizon (el=0°)."""
        direction = sun_direction_from_angles(270, 0)
        expected = np.array([-1.0, 0.0, 0.0])  # Pointing West
        np.testing.assert_array_almost_equal(direction, expected)

    def test_sun_zenith(self):
        """Sun at zenith (el=90°), azimuth doesn't matter."""
        direction = sun_direction_from_angles(0, 90)
        expected = np.array([0.0, 0.0, 1.0])  # Pointing Up
        np.testing.assert_array_almost_equal(direction, expected)

    def test_sun_nadir(self):
        """Sun at nadir (el=-90°), azimuth doesn't matter."""
        direction = sun_direction_from_angles(0, -90)
        expected = np.array([0.0, 0.0, -1.0])  # Pointing Down
        np.testing.assert_array_almost_equal(direction, expected)

    def test_sun_north_45_elevation(self):
        """Sun in North (az=0°), 45° elevation."""
        direction = sun_direction_from_angles(0, 45)
        sqrt2_2 = math.sqrt(2) / 2
        expected = np.array([0.0, sqrt2_2, sqrt2_2])
        np.testing.assert_array_almost_equal(direction, expected)

    def test_direction_is_unit_vector(self):
        """Direction vector should always be unit length."""
        for az in range(0, 360, 30):
            for el in range(-90, 91, 15):
                direction = sun_direction_from_angles(az, el)
                length = np.linalg.norm(direction)
                assert abs(length - 1.0) < 1e-10, f"Non-unit vector at az={az}, el={el}"


class TestAnglesFromSunDirection:
    """Tests for angles_from_sun_direction function."""

    def test_roundtrip_conversion(self):
        """Converting to direction and back should give same angles."""
        test_cases = [
            (0, 0),
            (90, 0),
            (180, 0),
            (270, 0),
            (45, 30),
            (135, 60),
            (225, 45),
            (315, 15),
        ]

        for az, el in test_cases:
            direction = sun_direction_from_angles(az, el)
            az_back, el_back = angles_from_sun_direction(direction)
            assert abs(az_back - az) < 1e-10 or abs(abs(az_back - az) - 360) < 1e-10
            assert abs(el_back - el) < 1e-10


class TestNormalize:
    """Tests for normalize function."""

    def test_normalize_simple(self):
        """Normalizing a simple vector."""
        v = np.array([3.0, 4.0, 0.0])
        result = normalize(v)
        expected = np.array([0.6, 0.8, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_already_unit(self):
        """Normalizing a unit vector returns same vector."""
        v = np.array([1.0, 0.0, 0.0])
        result = normalize(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_normalize_zero_raises(self):
        """Normalizing zero vector should raise."""
        v = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            normalize(v)


class TestAngleBetweenVectors:
    """Tests for angle_between_vectors function."""

    def test_parallel_vectors(self):
        """Parallel vectors have 0° angle."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([2.0, 0.0, 0.0])
        assert abs(angle_between_vectors(a, b) - 0.0) < 1e-10

    def test_perpendicular_vectors(self):
        """Perpendicular vectors have 90° angle."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(angle_between_vectors(a, b) - 90.0) < 1e-10

    def test_opposite_vectors(self):
        """Opposite vectors have 180° angle."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert abs(angle_between_vectors(a, b) - 180.0) < 1e-10

    def test_45_degree_angle(self):
        """Vectors at 45° angle."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 1.0, 0.0])
        assert abs(angle_between_vectors(a, b) - 45.0) < 1e-10
