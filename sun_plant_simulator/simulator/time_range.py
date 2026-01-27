"""Time-range simulation for running hit tests over multiple timestamps.

This module provides functionality to process sun position data over a time range
and output hit intervals.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.hit_test import check_sun_hits_plant
from ..core.models import Config, HitResult


@dataclass
class SunDataPoint:
    """A single sun position data point.

    Attributes:
        timestamp: ISO format timestamp string.
        azimuth_deg: Sun azimuth in degrees [0, 360).
        elevation_deg: Sun elevation in degrees [-90, +90].
    """

    timestamp: str
    azimuth_deg: float
    elevation_deg: float

    @classmethod
    def from_dict(cls, data: dict) -> "SunDataPoint":
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            azimuth_deg=data["azimuth_deg"],
            elevation_deg=data["elevation_deg"],
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
        }


@dataclass
class TimestampResult:
    """Result for a single timestamp.

    Attributes:
        timestamp: The timestamp string.
        hit_result: The HitResult from the hit test.
    """

    timestamp: str
    hit_result: HitResult


@dataclass
class HitInterval:
    """A continuous interval where the plant is hit by sunlight.

    Attributes:
        start_timestamp: Start of the interval.
        end_timestamp: End of the interval.
        window_id: ID of the window through which sunlight passes.
        n_timestamps: Number of timestamps in this interval.
    """

    start_timestamp: str
    end_timestamp: str
    window_id: Optional[str]
    n_timestamps: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "start": self.start_timestamp,
            "end": self.end_timestamp,
            "window_id": self.window_id,
            "n_timestamps": self.n_timestamps,
        }


@dataclass
class SimulationResult:
    """Result of a time-range simulation.

    Attributes:
        total_timestamps: Total number of timestamps processed.
        hit_count: Number of timestamps where the plant was hit.
        miss_count: Number of timestamps where the plant was not hit.
        hit_intervals: List of continuous hit intervals.
        results: Per-timestamp results (optional, for detailed analysis).
    """

    total_timestamps: int
    hit_count: int
    miss_count: int
    hit_intervals: list[HitInterval]
    results: list[TimestampResult] = field(default_factory=list)

    @property
    def hit_percentage(self) -> float:
        """Percentage of timestamps with hits."""
        if self.total_timestamps == 0:
            return 0.0
        return 100.0 * self.hit_count / self.total_timestamps

    def to_dict(self, include_details: bool = False) -> dict:
        """Convert to dictionary."""
        result = {
            "total_timestamps": self.total_timestamps,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_percentage": round(self.hit_percentage, 2),
            "hit_intervals": [i.to_dict() for i in self.hit_intervals],
        }
        if include_details:
            result["results"] = [
                {
                    "timestamp": r.timestamp,
                    "is_hit": r.hit_result.is_hit,
                    "window_id": r.hit_result.window_id,
                    "reason": r.hit_result.reason,
                }
                for r in self.results
            ]
        return result


def consolidate_to_intervals(results: list[TimestampResult]) -> list[HitInterval]:
    """Convert a list of timestamp results into continuous hit intervals.

    Groups consecutive hit timestamps into intervals.

    Args:
        results: List of TimestampResult objects in chronological order.

    Returns:
        List of HitInterval objects.
    """
    if not results:
        return []

    intervals = []
    current_interval_start = None
    current_interval_window = None
    current_interval_count = 0

    for result in results:
        if result.hit_result.is_hit:
            if current_interval_start is None:
                # Start new interval
                current_interval_start = result.timestamp
                current_interval_window = result.hit_result.window_id
                current_interval_count = 1
            else:
                # Extend current interval
                current_interval_count += 1
        else:
            if current_interval_start is not None:
                # End current interval
                intervals.append(
                    HitInterval(
                        start_timestamp=current_interval_start,
                        end_timestamp=results[
                            results.index(result) - 1
                        ].timestamp,
                        window_id=current_interval_window,
                        n_timestamps=current_interval_count,
                    )
                )
                current_interval_start = None
                current_interval_window = None
                current_interval_count = 0

    # Handle final interval if simulation ends with a hit
    if current_interval_start is not None:
        intervals.append(
            HitInterval(
                start_timestamp=current_interval_start,
                end_timestamp=results[-1].timestamp,
                window_id=current_interval_window,
                n_timestamps=current_interval_count,
            )
        )

    return intervals


def simulate_time_range(
    sun_data: list[SunDataPoint],
    config: Config,
    keep_details: bool = True,
) -> SimulationResult:
    """Run simulation over a time range of sun positions.

    Processes each sun data point and determines if the plant is hit.
    Groups results into continuous hit intervals.

    Args:
        sun_data: List of sun position data points.
        config: Configuration with plant and window geometry.
        keep_details: Whether to keep per-timestamp results.

    Returns:
        SimulationResult with hit counts, intervals, and optionally per-timestamp details.
    """
    results = []

    for point in sun_data:
        hit = check_sun_hits_plant(
            sun_azimuth_deg=point.azimuth_deg,
            sun_elevation_deg=point.elevation_deg,
            plant=config.plant,
            windows=config.windows,
            n_angular=config.simulation.sample_points_angular,
            n_vertical=config.simulation.sample_points_vertical,
        )
        results.append(TimestampResult(timestamp=point.timestamp, hit_result=hit))

    # Consolidate into intervals
    intervals = consolidate_to_intervals(results)

    hit_count = sum(1 for r in results if r.hit_result.is_hit)

    return SimulationResult(
        total_timestamps=len(results),
        hit_count=hit_count,
        miss_count=len(results) - hit_count,
        hit_intervals=intervals,
        results=results if keep_details else [],
    )


def load_sun_data_from_json(path: str | Path) -> list[SunDataPoint]:
    """Load sun position data from a JSON file.

    Expected format:
    {
        "data": [
            {"timestamp": "2024-01-15T08:00:00", "azimuth_deg": 120, "elevation_deg": 15},
            ...
        ]
    }

    Or simply an array:
    [
        {"timestamp": "2024-01-15T08:00:00", "azimuth_deg": 120, "elevation_deg": 15},
        ...
    ]

    Args:
        path: Path to the JSON file.

    Returns:
        List of SunDataPoint objects.
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Handle both formats
    if isinstance(data, list):
        points = data
    elif isinstance(data, dict) and "data" in data:
        points = data["data"]
    else:
        raise ValueError("Invalid sun data format")

    return [SunDataPoint.from_dict(p) for p in points]


def save_simulation_result(
    result: SimulationResult,
    path: str | Path,
    include_details: bool = False,
) -> None:
    """Save simulation result to a JSON file.

    Args:
        result: The simulation result.
        path: Output file path.
        include_details: Whether to include per-timestamp details.
    """
    with open(path, "w") as f:
        json.dump(result.to_dict(include_details), f, indent=2)
