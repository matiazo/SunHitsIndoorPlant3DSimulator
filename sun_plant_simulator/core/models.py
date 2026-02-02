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
        axis: Which room axis the wall runs along ("x" or "y" in simplified coords).
        position_along_wall: Distance from the shared corner to the window's inner edge
            measured along the wall axis. Stored so APIs can round-trip user input positions.
    """

    id: str
    center: np.ndarray
    width: float
    height: float
    wall_normal_azimuth: float
    wall_id: Optional[str] = None
    wall_thickness: float = 0.0
    axis: Optional[str] = None
    position_along_wall: Optional[float] = None

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
    """A wall definition including visualization metadata."""

    id: str
    outward_normal_azimuth_deg: float
    draw_length: float = 15.0


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
        viz_defaults = data.get("visualization", {})
        global_wall_length = viz_defaults.get("wall_length")

        walls: list[Wall] = []
        wall_props: dict[str, dict[str, float | str | None]] = {}
        for w in data.get("walls", []):
            wall_id = w["id"]
            wall_normal = float(w["outward_normal_azimuth_deg"])
            wall_thickness = float(w.get("thickness", 0.0) or 0.0)
            wall_axis = w.get("axis")

            viz_block = w.get("visualization", {}) or {}
            specific_length = (
                viz_block.get("wall_length")
                or viz_block.get("draw_length")
                or w.get("draw_length")
                or w.get("wall_length")
                or global_wall_length
            )
            draw_length = float(specific_length) if specific_length is not None else 15.0

            walls.append(
                Wall(
                    id=wall_id,
                    outward_normal_azimuth_deg=wall_normal,
                    draw_length=draw_length,
                )
            )

            wall_props[wall_id] = {
                "normal": wall_normal,
                "thickness": wall_thickness,
                "axis": wall_axis,
                "draw_length": draw_length,
            }

        corner_data = data.get("corner", {})
        corner_x = corner_data.get("x", 0.0)
        corner_y = corner_data.get("y", 0.0)

        def _infer_axis(window_dict: dict, window_wall_id: Optional[str]) -> Optional[str]:
            axis = window_dict.get("axis")
            if axis in {"x", "y"}:
                return axis
            if window_wall_id and window_wall_id in wall_props:
                candidate = wall_props[window_wall_id].get("axis")
                if candidate in {"x", "y"}:
                    return candidate
            center_val = window_dict.get("center")
            if center_val:
                # Windows on wall 1 (axis x) have |y| smaller than |x| in simplified coords
                return "x" if abs(center_val[1]) <= abs(center_val[0]) else "y"
            return None

        def _derive_position(window_dict: dict, axis: Optional[str], center_val: Optional[list | tuple]) -> Optional[float]:
            if axis == "x":
                if window_dict.get("position_along_wall") is not None:
                    return float(window_dict["position_along_wall"])
                if window_dict.get("x_position") is not None:
                    return float(window_dict["x_position"])
                if center_val is not None:
                    return float(center_val[0]) - float(window_dict["width"]) / 2 - corner_x
            elif axis == "y":
                if window_dict.get("position_along_wall") is not None:
                    return float(window_dict["position_along_wall"])
                if window_dict.get("y_position") is not None:
                    return float(window_dict["y_position"])
                if center_val is not None:
                    return float(center_val[1]) - float(window_dict["width"]) / 2 - corner_y
            return None

        def _derive_center_z(window_dict: dict, center_val: Optional[list | tuple]) -> float:
            z_bottom = window_dict.get("z_bottom")
            z_top = window_dict.get("z_top")
            if z_bottom is not None and z_top is not None:
                return float(z_bottom + z_top) / 2.0
            if center_val is not None and len(center_val) == 3:
                return float(center_val[2])
            raise ValueError(
                f"Window {window_dict.get('id', '<unknown>')} must define z_bottom/z_top or center[2]"
            )

        windows = []
        for w in data.get("windows", []):
            if "width" not in w or "height" not in w:
                raise ValueError(f"Window {w.get('id', '<unknown>')} missing width/height")

            width = float(w["width"])
            height = float(w["height"])

            wall_id = w.get("wall_id")
            wall_normal = w.get("wall_normal_azimuth_deg")
            wall_thickness = w.get("wall_thickness")

            wall_info = wall_props.get(wall_id, {})
            if wall_normal is None:
                wall_normal = wall_info.get("normal")
            if wall_thickness is None or wall_thickness == 0.0:
                wall_thickness = wall_info.get("thickness", 0.0)

            if wall_normal is None:
                raise ValueError(
                    f"Window {w.get('id', '<unknown>')} missing wall_normal_azimuth information"
                )

            axis = _infer_axis(w, wall_id)
            center_from_config = w.get("center")
            position_along_wall = _derive_position(w, axis, center_from_config)
            center_z = _derive_center_z(w, center_from_config)

            if axis in {"x", "y"} and position_along_wall is not None:
                if axis == "x":
                    center_x = corner_x + position_along_wall + width / 2
                    center_y = corner_y
                else:
                    center_y = corner_y + position_along_wall + width / 2
                    center_x = corner_x
            elif center_from_config is not None:
                center_x = float(center_from_config[0])
                center_y = float(center_from_config[1])
                if axis == "x" and position_along_wall is None:
                    position_along_wall = center_x - corner_x - width / 2
                elif axis == "y" and position_along_wall is None:
                    position_along_wall = center_y - corner_y - width / 2
            else:
                raise ValueError(
                    f"Window {w.get('id', '<unknown>')} must provide either center or axis-aligned position"
                )

            window_center = np.array([center_x, center_y, center_z], dtype=float)

            windows.append(
                Window(
                    id=w["id"],
                    center=window_center,
                    width=width,
                    height=height,
                    wall_normal_azimuth=wall_normal,
                    wall_id=wall_id,
                    wall_thickness=wall_thickness or 0.0,
                    axis=axis,
                    position_along_wall=position_along_wall,
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
                {
                    "id": w.id,
                    "outward_normal_azimuth_deg": w.outward_normal_azimuth_deg,
                    "visualization": {"wall_length": w.draw_length},
                }
                for w in self.walls
            ],
            "windows": [
                self._window_to_dict(w)
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

    @staticmethod
    def _window_to_dict(window: Window) -> dict:
        data = {
            "id": window.id,
            "wall_id": window.wall_id,
            "center": window.center.tolist(),
            "width": window.width,
            "height": window.height,
            "wall_normal_azimuth_deg": window.wall_normal_azimuth,
            "wall_thickness": window.wall_thickness,
        }

        if window.axis:
            data["axis"] = window.axis

        if window.position_along_wall is not None:
            if window.axis == "x":
                data["x_position"] = window.position_along_wall
            elif window.axis == "y":
                data["y_position"] = window.position_along_wall
            else:
                data["position_along_wall"] = window.position_along_wall

        return data
