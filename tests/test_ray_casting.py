"""Tests for the ray casting module."""

import numpy as np
import pytest

from sun_plant_simulator.core.models import Window
from sun_plant_simulator.core.ray_casting import (
    ray_intersects_window,
    ray_window_intersection,
)


def create_test_window(
    center=(0, 0, 5),
    width=2.0,
    height=2.0,
    wall_normal_azimuth=180,  # South-facing wall (normal points South)
) -> Window:
    """Create a window for testing."""
    return Window(
        id="test_window",
        center=np.array(center, dtype=float),
        width=width,
        height=height,
        wall_normal_azimuth=wall_normal_azimuth,
    )


class TestRayIntersectsWindow:
    """Tests for ray_intersects_window function."""

    def test_direct_hit_center(self):
        """Ray pointing directly at window center should hit."""
        # Window on south wall, normal pointing south (az=180)
        window = create_test_window(center=(0, 5, 5), wall_normal_azimuth=180)

        # Ray from inside room (y=0) pointing south toward window
        origin = np.array([0, 0, 5])
        direction = np.array([0, 1, 0])  # Pointing north (toward window from inside)

        # This should NOT hit because direction is pointing INTO room
        # (opposite to normal direction)
        assert not ray_intersects_window(origin, direction, window)

    def test_ray_through_window_from_inside(self):
        """Ray from inside pointing outward through window should hit."""
        # Window on south wall (y=5), normal pointing south (az=180)
        # Normal direction: (0, -1, 0) pointing toward negative y
        window = create_test_window(center=(0, 5, 5), wall_normal_azimuth=180)

        # Ray from inside (y=3) pointing south (same direction as normal)
        origin = np.array([0, 3, 5])
        direction = np.array([0, -1, 0])  # Pointing south (outward)

        # This SHOULD hit - ray going outward through window
        # Wait, let me reconsider. The normal for az=180 is:
        # sin(180) = 0, cos(180) = -1, so normal = (0, -1, 0)
        # Ray direction (0, -1, 0) has positive dot with normal (0, -1, 0)
        # So ray is going in same direction as normal (outward)

        # But the ray origin is at y=3, window is at y=5
        # So ray (y=3) + t*(0,-1,0) hits plane at y=5?
        # 3 - t = 5 → t = -2 (negative, behind origin)
        # This should NOT hit because t < 0

        # Let me fix the test - ray should be on OTHER side of window
        origin = np.array([0, 7, 5])  # Outside the room (south of window)
        direction = np.array([0, -1, 0])  # Still pointing south

        # Now: 7 - t = 5 → t = 2 (positive!)
        # But wait, direction and normal have same direction...
        # denom = dot((0,-1,0), (0,-1,0)) = 1 > 0
        # Ray IS going outward... but we want rays going FROM plant TO sun
        # The ray origin should be INSIDE (plant side)

        # Let me think again about the geometry:
        # - Window is at y=5
        # - Wall normal azimuth=180 means wall faces SOUTH (outward normal points south)
        # - So INSIDE the room is y < 5, OUTSIDE is y > 5
        # - Sunlight comes from OUTSIDE, hits plant INSIDE
        # - We cast ray FROM plant (inside) TOWARD sun (outside)
        # - Ray should go from y<5 toward y>5 (northward in this case, but...)

        # Actually the sun could be in ANY direction. If sun is in the south,
        # sun direction vector points south. But for light to come THROUGH the window,
        # the sun must be on the OUTSIDE of the wall.

        # Let me redo this test more carefully:
        # Window faces south (normal points south = negative y)
        # Sun in south means sun direction from plant is (0, -1, 0) approximately
        # Plant is inside at y=3
        # For ray to go outward (toward sun), it needs positive dot with normal

        origin = np.array([0, 3, 5])
        direction = np.array([0, -1, 0])  # Toward south

        # t = (window.center - origin) · normal / (direction · normal)
        # = ((0,5,5) - (0,3,5)) · (0,-1,0) / ((0,-1,0) · (0,-1,0))
        # = (0, 2, 0) · (0,-1,0) / 1
        # = -2 / 1 = -2

        # t is negative, so intersection is BEHIND origin. No hit.
        assert not ray_intersects_window(origin, direction, window)

    def test_ray_passing_through_window(self):
        """Test a ray that correctly passes through window."""
        # South-facing window at y=5
        # Normal = (sin(180), cos(180), 0) = (0, -1, 0)
        window = create_test_window(center=(0, 5, 5), wall_normal_azimuth=180)

        # Plant inside at y=3, shooting ray toward sun in south (y decreasing is south...)
        # Wait, I need to reconsider coordinate system.
        # y = North, so y increasing = north, y decreasing = south
        # Sun in south → direction from plant toward sun has negative y component

        # For light to enter through south-facing window:
        # - Window normal points south (into negative y)
        # - Sun is in south (negative y direction from room)
        # - Ray from plant toward sun goes toward negative y

        # Let's put plant at y=3 (inside), window at y=5, sun in direction (0, -1, 0)
        # But wait, if window is at y=5 and plant at y=3, ray going toward -y
        # would go AWAY from window (toward y=2, y=1, etc.)

        # The issue is I set up the geometry wrong. Let me fix:
        # South-facing window means the wall is on the SOUTH side of the room
        # So window should be at LOWER y values if room center is at higher y

        # Actually let's just make a simpler test:
        # Window at origin, facing north (normal = (0, 1, 0))
        window = Window(
            id="test",
            center=np.array([0, 0, 5]),
            width=2.0,
            height=2.0,
            wall_normal_azimuth=0,  # North-facing (normal points north)
        )
        # Normal = (sin(0), cos(0), 0) = (0, 1, 0)

        # Plant south of window at y=-3
        origin = np.array([0, -3, 5])
        # Ray toward sun in north (positive y)
        direction = np.array([0, 1, 0])

        # denom = (0,1,0)·(0,1,0) = 1 > 0 ✓
        # t = ((0,0,5)-(0,-3,5))·(0,1,0) / 1 = (0,3,0)·(0,1,0) = 3 > 0 ✓
        # intersection = (-3,0,5) + 3*(0,1,0) = (0, 0, 5) ✓ center of window
        # within bounds ✓

        assert ray_intersects_window(origin, direction, window)

    def test_ray_misses_window_horizontally(self):
        """Ray that passes to the side of window should miss."""
        window = Window(
            id="test",
            center=np.array([0, 0, 5]),
            width=2.0,
            height=2.0,
            wall_normal_azimuth=0,
        )

        # Plant offset to the side
        origin = np.array([5, -3, 5])  # 5m to the east
        direction = np.array([0, 1, 0])

        # Intersection would be at (5, 0, 5) which is outside window width
        assert not ray_intersects_window(origin, direction, window)

    def test_ray_misses_window_vertically(self):
        """Ray that passes above/below window should miss."""
        window = Window(
            id="test",
            center=np.array([0, 0, 5]),
            width=2.0,
            height=2.0,
            wall_normal_azimuth=0,
        )

        # Plant at floor level
        origin = np.array([0, -3, 1])  # z=1, well below window (z=4 to z=6)
        direction = np.array([0, 1, 0])

        # Intersection at (0, 0, 1) which is below window
        assert not ray_intersects_window(origin, direction, window)

    def test_ray_parallel_to_window(self):
        """Ray parallel to window plane should miss."""
        window = Window(
            id="test",
            center=np.array([0, 0, 5]),
            width=2.0,
            height=2.0,
            wall_normal_azimuth=0,
        )

        origin = np.array([0, -3, 5])
        direction = np.array([1, 0, 0])  # Parallel to window (east-west)

        assert not ray_intersects_window(origin, direction, window)

    def test_ray_pointing_into_room(self):
        """Ray pointing into room (opposite to normal) should miss."""
        window = Window(
            id="test",
            center=np.array([0, 0, 5]),
            width=2.0,
            height=2.0,
            wall_normal_azimuth=0,  # Normal points north (0, 1, 0)
        )

        # Plant north of window
        origin = np.array([0, 3, 5])
        direction = np.array([0, -1, 0])  # Pointing south (into room)

        # denom = (0,-1,0)·(0,1,0) = -1 < 0, so should return False
        assert not ray_intersects_window(origin, direction, window)


class TestRayWindowIntersection:
    """Tests for ray_window_intersection function (detailed results)."""

    def test_intersection_details(self):
        """Check that intersection details are correct."""
        window = Window(
            id="test",
            center=np.array([0, 0, 5]),
            width=2.0,
            height=2.0,
            wall_normal_azimuth=0,
        )

        origin = np.array([0, -3, 5])
        direction = np.array([0, 1, 0])

        result = ray_window_intersection(origin, direction, window)

        assert result.intersects
        assert result.t == pytest.approx(3.0)
        np.testing.assert_array_almost_equal(result.point, [0, 0, 5])
        assert result.local_h == pytest.approx(0.0)
        assert result.local_v == pytest.approx(0.0)

    def test_edge_of_window(self):
        """Test ray hitting edge of window."""
        window = Window(
            id="test",
            center=np.array([0, 0, 5]),
            width=2.0,
            height=2.0,
            wall_normal_azimuth=0,
        )

        # Hit right edge
        origin = np.array([1, -3, 5])  # x=1 = half width
        direction = np.array([0, 1, 0])

        result = ray_window_intersection(origin, direction, window)
        assert result.intersects
        assert abs(result.local_h) == pytest.approx(1.0)  # At edge (sign depends on axis direction)

        # Just outside right edge
        origin = np.array([1.1, -3, 5])
        result = ray_window_intersection(origin, direction, window)
        assert not result.intersects
