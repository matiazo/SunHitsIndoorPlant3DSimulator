"""Tests for config loading behavior related to window positions."""

import numpy as np
import pytest

from sun_plant_simulator.core.models import Config


def _base_config_dict() -> dict:
    return {
        "coordinate_system": "simplified",
        "corner": {"x": 0.0, "y": 0.0},
        "walls": [
            {"id": "wall_1", "outward_normal_azimuth_deg": 210, "axis": "x", "thickness": 0.3},
            {"id": "wall_2", "outward_normal_azimuth_deg": 300, "axis": "y", "thickness": 0.25},
        ],
        "plant": {"center_x": 5.0, "center_y": 4.0, "radius": 0.3, "z_min": 0.0, "z_max": 1.0},
        "windows": [],
    }


def test_window_centers_are_derived_from_positions():
    data = _base_config_dict()
    data["windows"] = [
        {
            "id": "wall1_window",
            "wall_id": "wall_1",
            "x_position": 2.0,
            "width": 1.0,
            "height": 1.5,
            "z_bottom": 4.0,
            "z_top": 5.0,
        },
        {
            "id": "wall2_window",
            "wall_id": "wall_2",
            "y_position": 3.0,
            "width": 0.8,
            "height": 1.2,
            "z_bottom": 1.0,
            "z_top": 2.2,
        },
    ]

    config = Config.from_dict(data)

    w1 = next(w for w in config.windows if w.id == "wall1_window")
    w2 = next(w for w in config.windows if w.id == "wall2_window")

    assert w1.axis == "x"
    assert w1.position_along_wall == pytest.approx(2.0)
    assert np.allclose(w1.center, [2.5, 0.0, 4.5])
    assert w1.wall_thickness == pytest.approx(0.3)

    assert w2.axis == "y"
    assert w2.position_along_wall == pytest.approx(3.0)
    # y center = corner_y + position + width/2 -> 0 + 3 + 0.4 = 3.4
    assert np.allclose(w2.center, [0.0, 3.4, 1.6])
    assert w2.wall_thickness == pytest.approx(0.25)


def test_legacy_centers_backfill_positions():
    data = _base_config_dict()
    data["windows"] = [
        {
            "id": "legacy",
            "wall_id": "wall_1",
            "width": 1.2,
            "height": 1.0,
            "center": [4.0, 0.0, 2.0],
            "z_bottom": 1.5,
            "z_top": 2.5,
        }
    ]

    config = Config.from_dict(data)
    legacy = config.windows[0]

    assert legacy.axis == "x"
    # center_x 4.0 -> position = center - width/2 = 4 - 0.6 = 3.4
    assert legacy.position_along_wall == pytest.approx(3.4)
    assert np.allclose(legacy.center, [4.0, 0.0, 2.0])


def test_wall_draw_length_defaults_to_15_meters():
    data = _base_config_dict()
    config = Config.from_dict(data)
    assert all(w.draw_length == pytest.approx(15.0) for w in config.walls)


def test_per_wall_draw_length_override():
    data = _base_config_dict()
    data["walls"][0]["visualization"] = {"wall_length": 18.5}
    data["walls"][1]["visualization"] = {"wall_length": 11.0}
    config = Config.from_dict(data)
    wall_lengths = {w.id: w.draw_length for w in config.walls}
    assert wall_lengths["wall_1"] == pytest.approx(18.5)
    assert wall_lengths["wall_2"] == pytest.approx(11.0)


def test_legacy_visualization_block_still_applies_globally():
    data = _base_config_dict()
    data["visualization"] = {"wall_length": 22.5}
    config = Config.from_dict(data)
    assert all(w.draw_length == pytest.approx(22.5) for w in config.walls)