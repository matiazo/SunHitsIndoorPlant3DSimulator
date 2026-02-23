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
"""

import logging
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE

from .const import (
    DOMAIN,
    CONF_UPDATE_INTERVAL,
    CONF_WALLS,
    CONF_WINDOWS,
    CONF_PLANT,
    DEFAULT_UPDATE_INTERVAL,
)

_LOGGER = logging.getLogger(__name__)

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


async def async_setup(hass: HomeAssistant, config: dict):
    """Set up the sun shade integration (YAML pass-through)."""
    return True


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
        from sun_plant_simulator.core.models import Config
        from sun_plant_simulator.core.window_sun import check_windows_from_config
    except ImportError as e:
        _LOGGER.error("Failed to import sun_plant_simulator: %s", e)
        _LOGGER.error(
            "Install via HACS or manually: pip install sun-plant-simulator"
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

    async def _async_update_data() -> dict[str, Any]:
        """Fetch sun data for all windows."""
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

        # Convert to a plain dict keyed by window_id
        window_data: dict[str, Any] = {}
        for window_id, detail in result.window_details.items():
            window_data[window_id] = {
                "is_in_sun": detail.is_in_sun,
                "intensity_factor": round(detail.intensity_factor, 3),
                "sun_angle_to_normal_deg": round(detail.sun_angle_to_normal_deg, 1),
            }

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
