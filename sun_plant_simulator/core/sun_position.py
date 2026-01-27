"""Sun position calculation based on location and time.

This module calculates the sun's azimuth and elevation for a given
latitude, longitude, date, and time using standard solar position algorithms.
"""

import math
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional


@dataclass
class SunPosition:
    """Sun position at a specific time.

    Attributes:
        azimuth_deg: Azimuth in degrees, clockwise from North [0, 360).
        elevation_deg: Elevation above horizon in degrees [-90, +90].
        timestamp: The datetime for this position.
    """
    azimuth_deg: float
    elevation_deg: float
    timestamp: datetime


@dataclass
class Location:
    """Geographic location.

    Attributes:
        latitude: Latitude in degrees (positive = North).
        longitude: Longitude in degrees (positive = East, negative = West).
        timezone_offset: Hours offset from UTC (e.g., -5 for EST, -4 for EDT).
    """
    latitude: float
    longitude: float
    timezone_offset: float = -5.0  # Default to EST

    @classmethod
    def from_config(cls, data: dict) -> "Location":
        """Create from config dictionary."""
        return cls(
            latitude=data["latitude"],
            longitude=data["longitude"],
            timezone_offset=data.get("timezone_offset", -5.0),
        )


def calculate_sun_position(
    latitude: float,
    longitude: float,
    dt: datetime,
    timezone_offset: float = 0.0,
) -> SunPosition:
    """Calculate sun position for a given location and time.

    Uses the NOAA solar position algorithm (simplified version).

    Args:
        latitude: Latitude in degrees (positive = North).
        longitude: Longitude in degrees (negative = West).
        dt: Local datetime.
        timezone_offset: Hours offset from UTC.

    Returns:
        SunPosition with azimuth and elevation.
    """
    # Convert to radians
    lat_rad = math.radians(latitude)

    # Day of year (1-366)
    day_of_year = dt.timetuple().tm_yday

    # Fractional year (radians)
    # For non-leap year
    days_in_year = 366 if _is_leap_year(dt.year) else 365
    gamma = 2 * math.pi / days_in_year * (day_of_year - 1 + (dt.hour - 12) / 24)

    # Equation of time (minutes)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )

    # Solar declination (radians)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    # Time offset (minutes)
    time_offset = eqtime + 4 * longitude - 60 * timezone_offset

    # True solar time (minutes)
    tst = dt.hour * 60 + dt.minute + dt.second / 60 + time_offset

    # Hour angle (degrees)
    ha = (tst / 4) - 180
    ha_rad = math.radians(ha)

    # Solar zenith angle
    cos_zenith = (
        math.sin(lat_rad) * math.sin(decl)
        + math.cos(lat_rad) * math.cos(decl) * math.cos(ha_rad)
    )
    cos_zenith = max(-1, min(1, cos_zenith))  # Clamp to [-1, 1]
    zenith_rad = math.acos(cos_zenith)

    # Solar elevation
    elevation_deg = 90 - math.degrees(zenith_rad)

    # Solar azimuth (measured clockwise from North)
    sin_zenith = math.sin(zenith_rad)
    if abs(sin_zenith) < 1e-10:
        azimuth_deg = 180.0  # Sun at zenith, arbitrarily set to South
    else:
        cos_azimuth = (
            (math.sin(lat_rad) * cos_zenith - math.sin(decl))
            / (math.cos(lat_rad) * sin_zenith)
        )
        cos_azimuth = max(-1, min(1, cos_azimuth))  # Clamp

        # This gives azimuth from South (0 = South, increases toward West)
        azimuth_from_south = math.degrees(math.acos(cos_azimuth))

        # Convert to azimuth from North (0 = North, 90 = East, 180 = South, 270 = West)
        if ha <= 0:
            # Morning: sun is in the East, azimuth should be < 180
            azimuth_deg = 180 - azimuth_from_south
        else:
            # Afternoon: sun is in the West, azimuth should be > 180
            azimuth_deg = 180 + azimuth_from_south

    return SunPosition(
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        timestamp=dt,
    )


def generate_sun_data_for_date(
    latitude: float,
    longitude: float,
    target_date: date,
    timezone_offset: float = -5.0,
    interval_minutes: int = 30,
    start_hour: int = 5,
    end_hour: int = 21,
) -> list[dict]:
    """Generate sun position data for an entire day.

    Args:
        latitude: Latitude in degrees.
        longitude: Longitude in degrees.
        target_date: The date to calculate for.
        timezone_offset: Hours offset from UTC.
        interval_minutes: Time between data points.
        start_hour: First hour to include (local time).
        end_hour: Last hour to include (local time).

    Returns:
        List of {timestamp, azimuth_deg, elevation_deg} dictionaries.
    """
    data = []

    current_time = datetime(target_date.year, target_date.month, target_date.day, start_hour, 0)
    end_time = datetime(target_date.year, target_date.month, target_date.day, end_hour, 0)

    while current_time <= end_time:
        pos = calculate_sun_position(latitude, longitude, current_time, timezone_offset)

        # Only include if sun is above horizon
        if pos.elevation_deg > -5:  # Include slightly below for twilight
            data.append({
                "timestamp": current_time.strftime("%H:%M"),
                "azimuth_deg": round(pos.azimuth_deg, 1),
                "elevation_deg": round(pos.elevation_deg, 1),
            })

        current_time += timedelta(minutes=interval_minutes)

    return data


def get_sunrise_sunset(
    latitude: float,
    longitude: float,
    target_date: date,
    timezone_offset: float = -5.0,
) -> tuple[Optional[datetime], Optional[datetime]]:
    """Get approximate sunrise and sunset times.

    Args:
        latitude: Latitude in degrees.
        longitude: Longitude in degrees.
        target_date: The date to calculate for.
        timezone_offset: Hours offset from UTC.

    Returns:
        Tuple of (sunrise_datetime, sunset_datetime). May be None for polar regions.
    """
    sunrise = None
    sunset = None

    # Search for sunrise (morning)
    for hour in range(4, 12):
        for minute in range(0, 60, 5):
            dt = datetime(target_date.year, target_date.month, target_date.day, hour, minute)
            pos = calculate_sun_position(latitude, longitude, dt, timezone_offset)
            if pos.elevation_deg > 0 and sunrise is None:
                sunrise = dt
                break
        if sunrise:
            break

    # Search for sunset (evening)
    for hour in range(20, 12, -1):
        for minute in range(55, -1, -5):
            dt = datetime(target_date.year, target_date.month, target_date.day, hour, minute)
            pos = calculate_sun_position(latitude, longitude, dt, timezone_offset)
            if pos.elevation_deg > 0 and sunset is None:
                sunset = dt
                break
        if sunset:
            break

    return sunrise, sunset


def _is_leap_year(year: int) -> bool:
    """Check if a year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
