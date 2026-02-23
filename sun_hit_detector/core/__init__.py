"""Core hit-test algorithm components."""

from .models import Window, Plant, HitResult, Config
from .geometry import sun_direction_from_angles
from .ray_casting import ray_intersects_window
from .hit_test import check_sun_hits_plant, generate_plant_sample_points
from .coordinates import (
    position_from_wall_distances,
    wall_distances_from_position,
    plant_position_from_wall_distances,
)

__all__ = [
    "Window",
    "Plant",
    "HitResult",
    "Config",
    "sun_direction_from_angles",
    "ray_intersects_window",
    "check_sun_hits_plant",
    "generate_plant_sample_points",
    "position_from_wall_distances",
    "wall_distances_from_position",
    "plant_position_from_wall_distances",
]
