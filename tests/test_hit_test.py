"""Tests for the hit test module."""

import numpy as np
import pytest

from sun_plant_simulator.core.hit_test import (
    check_sun_hits_plant,
    generate_plant_sample_points,
)
from sun_plant_simulator.core.models import Plant, Window


def create_test_plant(
    center_x=0, center_y=0, radius=0.3, z_min=0, z_max=1.5
) -> Plant:
    """Create a test plant."""
    return Plant(
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        z_min=z_min,
        z_max=z_max,
    )


def create_test_window(
    center, width=2.0, height=2.0, wall_normal_azimuth=0
) -> Window:
    """Create a test window."""
    return Window(
        id="test_window",
        center=np.array(center, dtype=float),
        width=width,
        height=height,
        wall_normal_azimuth=wall_normal_azimuth,
    )


class TestGeneratePlantSamplePoints:
    """Tests for generate_plant_sample_points function."""

    def test_point_count(self):
        """Check correct number of sample points generated."""
        plant = create_test_plant()

        # Default: 8 angular × 3 vertical + 2 center points = 26
        points = generate_plant_sample_points(plant, n_angular=8, n_vertical=3)
        assert len(points) == 8 * 3 + 2

        # Custom: 4 angular × 2 vertical + 2 = 10
        points = generate_plant_sample_points(plant, n_angular=4, n_vertical=2)
        assert len(points) == 4 * 2 + 2

    def test_points_on_cylinder_surface(self):
        """Sample points should be on the cylinder surface."""
        plant = create_test_plant(center_x=5, center_y=3, radius=0.5)
        points = generate_plant_sample_points(plant, n_angular=8, n_vertical=3)

        # Check surface points (exclude center points at end)
        for point in points[:-2]:
            # Distance from axis should equal radius
            dist = np.sqrt(
                (point[0] - plant.center_x) ** 2 + (point[1] - plant.center_y) ** 2
            )
            assert dist == pytest.approx(plant.radius, rel=1e-6)

            # Z should be within plant height
            assert plant.z_min <= point[2] <= plant.z_max

    def test_center_points(self):
        """Check center points at top and middle."""
        plant = create_test_plant(center_x=2, center_y=3, z_min=0, z_max=2)
        points = generate_plant_sample_points(plant)

        # Last two points are center points
        top_center = points[-2]
        mid_center = points[-1]

        assert top_center[0] == pytest.approx(plant.center_x)
        assert top_center[1] == pytest.approx(plant.center_y)
        assert top_center[2] == pytest.approx(plant.z_max)

        assert mid_center[0] == pytest.approx(plant.center_x)
        assert mid_center[1] == pytest.approx(plant.center_y)
        assert mid_center[2] == pytest.approx((plant.z_min + plant.z_max) / 2)


class TestCheckSunHitsPlant:
    """Tests for check_sun_hits_plant function."""

    def test_sun_below_horizon(self):
        """Sun below horizon should never hit."""
        plant = create_test_plant()
        windows = [create_test_window(center=(0, 5, 5))]

        result = check_sun_hits_plant(
            sun_azimuth_deg=180,
            sun_elevation_deg=-10,  # Below horizon
            plant=plant,
            windows=windows,
        )

        assert not result.is_hit
        assert result.reason == "sun_below_horizon"

    def test_sun_at_horizon(self):
        """Sun exactly at horizon (el=0) should not hit."""
        plant = create_test_plant()
        windows = [create_test_window(center=(0, 5, 5))]

        result = check_sun_hits_plant(
            sun_azimuth_deg=180,
            sun_elevation_deg=0,  # Exactly at horizon
            plant=plant,
            windows=windows,
        )

        assert not result.is_hit
        assert result.reason == "sun_below_horizon"

    def test_basic_hit_scenario(self):
        """Test a basic scenario where sun should hit plant through window.

        Uses the axis-aligned coordinate system where:
        - Wall 1 is at y=0 (runs along x-axis)
        - Wall 2 is at x=0 (runs along y-axis)
        """
        # Plant inside the room
        plant = create_test_plant(center_x=5, center_y=3, radius=0.3, z_min=0, z_max=1.5)

        # Window on wall 1 (y=0), south-facing (normal points south, -y direction)
        window = create_test_window(
            center=(5, 0, 1.0),  # On wall 1 at y=0, aligned with plant x
            width=2.0,
            height=2.0,
            wall_normal_azimuth=180,  # South-facing (normal points -Y)
        )

        # Sun in the south - rays travel from south toward north through window
        result = check_sun_hits_plant(
            sun_azimuth_deg=180,  # South
            sun_elevation_deg=20,  # Above horizon
            plant=plant,
            windows=[window],
            wall1_normal_azimuth=180,  # Match window's wall normal
        )

        # Should hit - sun rays from south pass through south-facing window to plant inside
        assert result.is_hit
        assert result.window_id == "test_window"
        assert len(result.hit_points) > 0

    def test_no_window_path(self):
        """Test scenario where sun is on wrong side (no window path)."""
        plant = create_test_plant(center_x=5, center_y=3)

        # South-facing window on wall 1 (y=0)
        window = create_test_window(
            center=(5, 0, 1.0),
            wall_normal_azimuth=180,  # South-facing
        )

        # Sun in the north - opposite side from where window faces
        result = check_sun_hits_plant(
            sun_azimuth_deg=0,  # North
            sun_elevation_deg=45,
            plant=plant,
            windows=[window],
        )

        # Should NOT hit - sun is north but window faces south
        assert not result.is_hit
        assert result.reason == "no_window_path"

    def test_sun_direction_returned(self):
        """Check that sun direction is returned in result."""
        plant = create_test_plant(center_x=5, center_y=3)
        windows = [create_test_window(center=(5, 0, 1.0), wall_normal_azimuth=180)]

        result = check_sun_hits_plant(
            sun_azimuth_deg=45,
            sun_elevation_deg=30,
            plant=plant,
            windows=windows,
        )

        assert result.sun_direction is not None
        assert len(result.sun_direction) == 3
        # Should be approximately unit vector
        assert np.linalg.norm(result.sun_direction) == pytest.approx(1.0)


class TestMultipleWindows:
    """Tests with multiple windows.

    Uses axis-aligned coordinate system:
    - Wall 1 at y=0 (windows have center[1] ≈ 0)
    - Wall 2 at x=0 (windows have center[0] ≈ 0)
    """

    def test_hit_through_first_window(self):
        """Hit detected through first matching window."""
        # Plant inside the room
        plant = create_test_plant(center_x=3, center_y=3)

        windows = [
            Window(
                id="window_south",
                center=np.array([3, 0, 1.0]),  # Wall 1 (y=0), south-facing
                width=2.0,
                height=2.0,
                wall_normal_azimuth=180,  # South (normal -Y)
            ),
            Window(
                id="window_west",
                center=np.array([0, 3, 1.0]),  # Wall 2 (x=0), west-facing
                width=2.0,
                height=2.0,
                wall_normal_azimuth=270,  # West (normal -X)
            ),
        ]

        # Sun in south - should hit through south-facing window
        result = check_sun_hits_plant(
            sun_azimuth_deg=180,
            sun_elevation_deg=20,
            plant=plant,
            windows=windows,
            wall1_normal_azimuth=180,  # Match window_south wall normal
        )

        assert result.is_hit
        assert result.window_id == "window_south"

    def test_hit_through_second_window(self):
        """Hit detected through second window when first doesn't match."""
        # Plant inside the room
        plant = create_test_plant(center_x=3, center_y=3)

        windows = [
            Window(
                id="window_south",
                center=np.array([3, 0, 1.0]),  # Wall 1 (y=0), south-facing
                width=2.0,
                height=2.0,
                wall_normal_azimuth=180,  # South
            ),
            Window(
                id="window_west",
                center=np.array([0, 3, 1.0]),  # Wall 2 (x=0), west-facing
                width=2.0,
                height=2.0,
                wall_normal_azimuth=270,  # West (normal -X)
            ),
        ]

        # Sun in west - should hit through west-facing window
        result = check_sun_hits_plant(
            sun_azimuth_deg=270,
            sun_elevation_deg=20,
            plant=plant,
            windows=windows,
            wall1_normal_azimuth=180,  # Match window_south wall normal
        )

        assert result.is_hit
        assert result.window_id == "window_west"
