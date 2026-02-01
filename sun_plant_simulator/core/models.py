"""Data models for the sun-plant hit simulation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Location:
    """Geographic location for sun position calculations.

    Attributes:
        latitude: Latitude in degrees (positive = North).
        longitude: Longitude in degrees (positive = East).
        timezone_offset: Hours offset from UTC (e.g., -5 for EST).
        timezone_name: IANA timezone identifier (e.g., "America/New_York").
    """
    latitude: float
    longitude: float
    timezone_offset: float = -5.0
    timezone_name: Optional[str] = None


@dataclass
class Window:
    """A rectangular window on a wall.

    Attributes:
        id: Unique identifier for the window.
        center: Center point (x, y, z) in meters - this is the INNER wall face.
        width: Horizontal width in meters.
        height: Vertical height in meters.
        wall_normal_azimuth: Outward normal azimuth in degrees (clockwise from North).
        wall_id: Optional reference to parent wall.
        wall_thickness: Thickness of the wall in meters (0 = thin plane model).
    """

    id: str
    center: np.ndarray
    width: float
    height: float
    wall_normal_azimuth: float
    wall_id: Optional[str] = None
    wall_thickness: float = 0.0

    @property
    def normal(self) -> np.ndarray:
        """Outward-facing normal vector (horizontal, in xy plane)."""
        az_rad = math.radians(self.wall_normal_azimuth)
        return np.array([math.sin(az_rad), math.cos(az_rad), 0.0])

    @property
    def horizontal_axis(self) -> np.ndarray:
        """Horizontal axis along window (perpendicular to normal, in xy plane).

        This is the local "right" direction when facing the window from outside.
        """
        n = self.normal
        return np.array([-n[1], n[0], 0.0])

    @property
    def vertical_axis(self) -> np.ndarray:
        """Vertical axis (always pointing up)."""
        return np.array([0.0, 0.0, 1.0])

    @property
    def z_bottom(self) -> float:
        """Bottom edge z-coordinate."""
        return self.center[2] - self.height / 2

    @property
    def z_top(self) -> float:
        """Top edge z-coordinate."""
        return self.center[2] + self.height / 2

    def get_corners(self) -> list[np.ndarray]:
        """Get the four corner points of the window.

        Returns corners in order: bottom-left, bottom-right, top-right, top-left
        when viewed from outside (looking at normal).
        """
        h_half = self.width / 2
        v_half = self.height / 2
        h_axis = self.horizontal_axis
        v_axis = self.vertical_axis

        return [
            self.center - h_half * h_axis - v_half * v_axis,  # bottom-left
            self.center + h_half * h_axis - v_half * v_axis,  # bottom-right
            self.center + h_half * h_axis + v_half * v_axis,  # top-right
            self.center - h_half * h_axis + v_half * v_axis,  # top-left
        ]


@dataclass
class Plant:
    """A vertical cylinder representing a plant.

    Attributes:
        center_x: X-coordinate (East) of cylinder center in meters.
        center_y: Y-coordinate (North) of cylinder center in meters.
        radius: Radius of the cylinder in meters.
        z_min: Bottom of plant (typically 0 for floor).
        z_max: Top of plant in meters.
    """

    center_x: float
    center_y: float
    radius: float
    z_min: float
    z_max: float

    @property
    def center_xy(self) -> np.ndarray:
        """2D center point (x, y)."""
        return np.array([self.center_x, self.center_y])

    @property
    def height(self) -> float:
        """Total height of the plant."""
        return self.z_max - self.z_min


@dataclass
class Wall:
    """A wall definition (minimal - just the normal azimuth).

    Attributes:
        id: Unique identifier for the wall.
        outward_normal_azimuth_deg: Direction the wall faces (outward), in degrees.
    """

    id: str
    outward_normal_azimuth_deg: float


@dataclass
class HitResult:
    """Result of a sun-plant hit test.

    Attributes:
        is_hit: Whether the plant is hit by direct sunlight.
        window_id: ID of the window through which sunlight passes (if hit).
        hit_points: List of sample points that receive sunlight.
        sun_direction: Direction vector toward the sun.
        reason: Explanation if not hit (e.g., "sun_below_horizon").
    """

    is_hit: bool
    window_id: Optional[str] = None
    hit_points: list[np.ndarray] = field(default_factory=list)
    sun_direction: Optional[np.ndarray] = None
    reason: Optional[str] = None


@dataclass
class SimulationConfig:
    """Simulation parameters.

    Attributes:
        sample_points_angular: Number of angular divisions around cylinder.
        sample_points_vertical: Number of vertical divisions on cylinder.
    """

    sample_points_angular: int = 8
    sample_points_vertical: int = 3


@dataclass
class Config:
    """Complete configuration for the simulation.

    Attributes:
        walls: List of wall definitions.
        windows: List of window definitions.
        plant: Plant definition.
        simulation: Simulation parameters.
        location: Geographic location for sun calculations.
        coordinate_system: Coordinate system name (default "ENU").
        units: Units for measurements (default "meters").
    """

    walls: list[Wall]
    windows: list[Window]
    plant: Plant
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    location: Optional[Location] = None
    coordinate_system: str = "ENU"
    units: str = "meters"

    @classmethod
    def from_json_file(cls, path: str | Path) -> Config:
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        """Create configuration from a dictionary."""
        walls = [
            Wall(id=w["id"], outward_normal_azimuth_deg=w["outward_normal_azimuth_deg"])
            for w in data.get("walls", [])
        ]

        # Build a dict of wall properties for quick lookup
        wall_props = {}
        for w in data.get("walls", []):
            wall_props[w["id"]] = {
                "normal": w["outward_normal_azimuth_deg"],
                "thickness": w.get("thickness", 0.0),
                "axis": w.get("axis"),
            }

        windows = []
        for w in data.get("windows", []):
            # Find wall normal and thickness for this window
            wall_id = w.get("wall_id")
            wall_normal = w.get("wall_normal_azimuth_deg")
            wall_thickness = w.get("wall_thickness", 0.0)

            if wall_id and wall_id in wall_props:
                if wall_normal is None:
                    wall_normal = wall_props[wall_id]["normal"]
                if wall_thickness == 0.0:
                    wall_thickness = wall_props[wall_id]["thickness"]

            windows.append(
                Window(
                    id=w["id"],
                    center=np.array(w["center"], dtype=float),
                    width=w["width"],
                    height=w["height"],
                    wall_normal_azimuth=wall_normal,
                    wall_id=wall_id,
                    wall_thickness=wall_thickness,
                )
            )

        plant_data = data["plant"]

        # Support wall distances as alternative to x/y
        dist1 = plant_data.get("dist_from_wall1")
        dist2 = plant_data.get("dist_from_wall2")
        coordinate_system = data.get("coordinate_system", "ENU").lower()
        simplify_axes = coordinate_system == "simplified"

        if dist1 is not None and dist2 is not None:
            if simplify_axes:
                # In simplified mode, walls align with axes: wall_1 distance maps to y, wall_2 to x
                center_x = float(dist2)
                center_y = float(dist1)
            else:
                from .coordinates import position_from_wall_distances

                wall1_id = plant_data.get("wall1_id")
                wall2_id = plant_data.get("wall2_id")

                if wall1_id is None and walls:
                    wall1_id = walls[0].id
                if wall2_id is None and len(walls) > 1:
                    wall2_id = walls[1].id

                wall1_az = wall_props.get(wall1_id, {}).get("normal", 210)
                wall2_az = wall_props.get(wall2_id, {}).get("normal", 307)

                # Get corner position
                corner_data = data.get("corner", {})
                corner_x = corner_data.get("x", 0.0)
                corner_y = corner_data.get("y", 0.0)

                center_x, center_y = position_from_wall_distances(
                    dist_from_wall1=dist1,
                    dist_from_wall2=dist2,
                    wall1_normal_azimuth=wall1_az,
                    wall2_normal_azimuth=wall2_az,
                    corner_x=corner_x,
                    corner_y=corner_y,
                )
        else:
            center_x = plant_data.get("center_x")
            center_y = plant_data.get("center_y")

        if center_x is None or center_y is None:
            raise ValueError("Plant center coordinates could not be determined; provide center_x/center_y or wall distances")

        plant = Plant(
            center_x=center_x,
            center_y=center_y,
            radius=plant_data["radius"],
            z_min=plant_data["z_min"],
            z_max=plant_data["z_max"],
        )

        sim_data = data.get("simulation", {})
        simulation = SimulationConfig(
            sample_points_angular=sim_data.get("sample_points_angular", 8),
            sample_points_vertical=sim_data.get("sample_points_vertical", 3),
        )

        # Parse location data
        location_data = data.get("location", {})
        location = None
        if location_data:
            location = Location(
                latitude=location_data.get("latitude", 0.0),
                longitude=location_data.get("longitude", 0.0),
                timezone_offset=location_data.get("timezone_offset", -5.0),
                timezone_name=location_data.get("timezone_name"),
            )

        return cls(
            walls=walls,
            windows=windows,
            plant=plant,
            simulation=simulation,
            location=location,
            coordinate_system=data.get("coordinate_system", "ENU"),
            units=data.get("units", "meters"),
        )

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return {
            "coordinate_system": self.coordinate_system,
            "units": self.units,
            "walls": [
                {"id": w.id, "outward_normal_azimuth_deg": w.outward_normal_azimuth_deg}
                for w in self.walls
            ],
            "windows": [
                {
                    "id": w.id,
                    "wall_id": w.wall_id,
                    "center": w.center.tolist(),
                    "width": w.width,
                    "height": w.height,
                    "wall_normal_azimuth_deg": w.wall_normal_azimuth,
                }
                for w in self.windows
            ],
            "plant": {
                "center_x": self.plant.center_x,
                "center_y": self.plant.center_y,
                "radius": self.plant.radius,
                "z_min": self.plant.z_min,
                "z_max": self.plant.z_max,
            },
            "simulation": {
                "sample_points_angular": self.simulation.sample_points_angular,
                "sample_points_vertical": self.simulation.sample_points_vertical,
            },
        }
