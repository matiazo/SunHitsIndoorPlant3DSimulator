"""Sun Shade Integration for Home Assistant.

This custom component monitors sun position and creates sensor entities
with real-time window sun exposure data computed via 3D ray-casting simulation.

Setup via UI: Settings > Integrations > Add Integration > "Sun Shade Integration"

Room geometry (walls, windows, plant position) is configured entirely through the
HA config flow UI — no JSON file needed.

Created entities per window:
- binary_sensor.<window_id>_has_sun: Whether the window is receiving direct sun
- sensor.<window_id>_sun_intensity: Intensity factor (0.0-1.0)
- sensor.<window_id>_sun_angle: Angle between sun and window normal (0-90 degrees)

Plant-level entities (daily forecast via ray-casting):
- sensor.plant_sun_start: First time sun hits the plant today (timestamp)
- sensor.plant_sun_end: Last time sun hits the plant today (timestamp)
- sensor.plant_sun_duration: Total sun exposure duration today (minutes)
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    CONF_UPDATE_INTERVAL,
    CONF_WALLS,
    CONF_WINDOWS,
    CONF_PLANT,
    DEFAULT_UPDATE_INTERVAL,
)

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
PLATFORMS = ["binary_sensor", "sensor"]


def _build_config_dict(data: dict) -> dict:
    """Wrap entry.data into the format Config.from_dict() expects."""
    return {
        "coordinate_system": "ENU",
        "corner": {"x": 0.0, "y": 0.0},
        "walls": data.get(CONF_WALLS, []),
        "windows": data.get(CONF_WINDOWS, []),
        "plant": data.get(CONF_PLANT, {}),
        "simulation": {
            "sample_points_angular": 8,
            "sample_points_vertical": 3,
        },
    }


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old config entries.

    VERSION 1 entries used a JSON file path and cannot be auto-migrated.
    """
    if entry.version == 1:
        _LOGGER.error(
            "Config entry version 1 (JSON file based) cannot be auto-migrated. "
            "Please remove and re-add the Sun Shade Integration."
        )
        return False
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Sun Shade Integration from a config entry."""
    data = {**entry.data, **entry.options}
    update_interval = data.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)

    try:
        from sun_hit_detector.core.models import Config
        from sun_hit_detector.core.window_sun import check_windows_from_config
        from sun_hit_detector.core.sun_position import generate_sun_data_for_date
        from sun_hit_detector.core.hit_test import (
            check_sun_hits_plant_from_config,
            check_plant_hit_per_window_from_config,
        )
    except ImportError as e:
        _LOGGER.error("Failed to import sun_hit_detector: %s", e)
        _LOGGER.error(
            "Install via HACS or manually: pip install sun-hit-detector"
        )
        return False

    # Build Config from entry data
    try:
        config_dict = _build_config_dict(data)
        sim_config = await hass.async_add_executor_job(
            Config.from_dict, config_dict
        )
        _LOGGER.info(
            "Built sun simulator config from entry data: %d walls, %d windows",
            len(data.get(CONF_WALLS, [])),
            len(data.get(CONF_WINDOWS, [])),
        )
    except Exception:
        _LOGGER.exception("Failed to build Config from entry data")
        return False

    # Cache for daily plant forecast (recomputed only when date changes)
    _forecast_cache: dict[str, Any] = {"date": None, "data": None}

    def _compute_daily_plant_forecast(today: date) -> dict[str, Any]:
        """Compute plant sun forecast for the entire day.

        Scans sun positions at 15-minute intervals and uses the plant-level
        ray-casting hit test to determine when sunlight reaches the plant.

        Returns dict with sun_start, sun_end (HH:MM strings), and
        sun_duration_min (int minutes). All None if no sun hits the plant.
        """
        latitude = hass.config.latitude
        longitude = hass.config.longitude
        tz_name = hass.config.time_zone

        sun_data = generate_sun_data_for_date(
            latitude=latitude,
            longitude=longitude,
            target_date=today,
            timezone_name=tz_name,
            interval_minutes=15,
            start_hour=5,
            end_hour=21,
        )

        first_hit: str | None = None
        last_hit: str | None = None
        hit_count = 0

        for point in sun_data:
            hit_result = check_sun_hits_plant_from_config(
                sun_azimuth_deg=point["azimuth_deg"],
                sun_elevation_deg=point["elevation_deg"],
                config=sim_config,
            )
            if hit_result.is_hit:
                hit_count += 1
                if first_hit is None:
                    first_hit = point["timestamp"]
                last_hit = point["timestamp"]

        return {
            "sun_start": first_hit,
            "sun_end": last_hit,
            "sun_duration_min": hit_count * 15,
        }

    async def _async_update_data() -> dict[str, Any]:
        """Fetch sun data for all windows and daily plant forecast."""
        sun_entity = hass.states.get("sun.sun")
        if not sun_entity or sun_entity.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            raise UpdateFailed("Sun entity not available")

        azimuth = sun_entity.attributes.get("azimuth")
        elevation = sun_entity.attributes.get("elevation")

        if azimuth is None or elevation is None:
            raise UpdateFailed("Sun position attributes missing from sun.sun")

        try:
            result = await hass.async_add_executor_job(
                check_windows_from_config,
                azimuth,
                elevation,
                sim_config,
            )
        except Exception as err:
            raise UpdateFailed(f"Error computing sun exposure: {err}") from err

        # Per-window plant hit test: does sun through each window reach the plant?
        try:
            plant_hits = await hass.async_add_executor_job(
                check_plant_hit_per_window_from_config,
                azimuth,
                elevation,
                sim_config,
            )
        except Exception as err:
            raise UpdateFailed(f"Error computing plant hits: {err}") from err

        # Convert to a plain dict keyed by window_id
        window_data: dict[str, Any] = {}
        for window_id, detail in result.window_details.items():
            window_data[window_id] = {
                "is_in_sun": plant_hits.get(window_id, False),
                "intensity_factor": round(detail.intensity_factor, 3),
                "sun_angle_to_normal_deg": round(detail.sun_angle_to_normal_deg, 1),
            }

        # Compute daily plant forecast (cached per date)
        today = date.today()
        if _forecast_cache["date"] != today:
            try:
                forecast = await hass.async_add_executor_job(
                    _compute_daily_plant_forecast, today
                )
                _forecast_cache["date"] = today
                _forecast_cache["data"] = forecast
                _LOGGER.debug("Computed daily plant forecast for %s: %s", today, forecast)
            except Exception:
                _LOGGER.exception("Failed to compute daily plant forecast")
                _forecast_cache["data"] = {
                    "sun_start": None,
                    "sun_end": None,
                    "sun_duration_min": 0,
                }

        window_data["_plant_forecast"] = _forecast_cache["data"]

        _LOGGER.debug(
            "Sun update: azimuth=%.1f, elevation=%.1f, windows_in_sun=%s",
            azimuth,
            elevation,
            result.windows_in_sun,
        )
        return window_data

    coordinator = DataUpdateCoordinator(
        hass,
        _LOGGER,
        name=DOMAIN,
        update_method=_async_update_data,
        update_interval=timedelta(seconds=update_interval),
    )

    # Do first refresh
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    _LOGGER.info(
        "Sun shade integration initialized. Updates every %d seconds.",
        update_interval,
    )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
        if not hass.data[DOMAIN]:
            hass.data.pop(DOMAIN)

    return unload_ok
