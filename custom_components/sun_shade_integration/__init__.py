"""Sun Shade Integration for Home Assistant.

This custom component monitors sun position and updates shade entity attributes
with window sun exposure information. It adds attributes to existing shade entities
to indicate whether their associated windows are receiving direct sunlight.

Configuration in configuration.yaml:
    sun_shade_integration:
      config_path: /config/sun_plant_config.json
      update_interval: 300  # seconds (5 minutes)

The component reads window-to-shade mappings from the config file and updates
shade entities with the following attributes:
- window_id: The window associated with this shade
- window_has_sun: Boolean indicating if window is receiving direct sun
- sun_intensity: Intensity factor (0.0-1.0) representing relative sun intensity
- sun_angle_deg: Angle between sun and window normal (0-90 degrees)
"""

import logging
from datetime import timedelta
from pathlib import Path

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE

from .const import (
    DOMAIN,
    CONF_CONFIG_PATH,
    CONF_UPDATE_INTERVAL,
    DEFAULT_UPDATE_INTERVAL,
    ATTR_WINDOW_ID,
    ATTR_WINDOW_HAS_SUN,
    ATTR_SUN_INTENSITY,
    ATTR_SUN_ANGLE,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: dict):
    """Set up the sun shade integration.

    Args:
        hass: Home Assistant instance
        config: Configuration dictionary from configuration.yaml

    Returns:
        True if setup was successful, False otherwise
    """
    domain_config = config.get(DOMAIN, {})
    config_path = domain_config.get(CONF_CONFIG_PATH, "/config/sun_plant_config.json")
    update_interval = domain_config.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)

    # Import sun simulator modules
    import sys
    sys.path.insert(0, "/sun-plant-simulator")

    try:
        from sun_plant_simulator.core.models import Config
        from sun_plant_simulator.core.window_sun import check_windows_from_config
    except ImportError as e:
        _LOGGER.error(f"Failed to import sun simulator modules: {e}")
        _LOGGER.error("Make sure /sun-plant-simulator is mounted correctly")
        return False

    # Load config
    try:
        sim_config = Config.from_json_file(config_path)
        _LOGGER.info(f"Loaded sun simulator config from {config_path}")
    except Exception as e:
        _LOGGER.error(f"Failed to load config from {config_path}: {e}")
        return False

    # Build window → shade mapping
    window_to_shade = {}
    for window in sim_config.windows:
        if window.shade_entity_id:
            window_to_shade[window.id] = window.shade_entity_id

    _LOGGER.info(f"Found {len(window_to_shade)} windows with shade mappings")

    if not window_to_shade:
        _LOGGER.warning("No windows with shade_entity_id found in config")
        _LOGGER.warning("Add shade_entity_id to window definitions in config file")

    @callback
    async def update_shade_sun_attributes(now=None):
        """Update sun attributes on shade entities.

        This function is called periodically (default: every 5 minutes) to:
        1. Get current sun position from sun.sun entity
        2. Calculate which windows are receiving direct sunlight
        3. Update shade entity attributes with sun exposure data

        Args:
            now: Current time (provided by async_track_time_interval)
        """
        # Get sun position from sun.sun entity
        sun_entity = hass.states.get("sun.sun")
        if not sun_entity or sun_entity.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            _LOGGER.warning("Sun entity not available")
            return

        azimuth = sun_entity.attributes.get("azimuth")
        elevation = sun_entity.attributes.get("elevation")

        if azimuth is None or elevation is None:
            _LOGGER.warning("Sun position attributes missing from sun.sun entity")
            return

        # Run window sun calculation
        try:
            result = await hass.async_add_executor_job(
                check_windows_from_config,
                azimuth,
                elevation,
                sim_config,
            )

            # Update shade entity attributes
            updated_count = 0
            for window_id, shade_entity_id in window_to_shade.items():
                window_detail = result.window_details.get(window_id)
                if not window_detail:
                    _LOGGER.warning(f"No window detail found for {window_id}")
                    continue

                # Get current shade entity state
                shade_entity = hass.states.get(shade_entity_id)
                if not shade_entity:
                    _LOGGER.warning(f"Shade entity {shade_entity_id} not found")
                    continue

                # Merge existing attributes with new window sun attributes
                new_attributes = dict(shade_entity.attributes)
                new_attributes[ATTR_WINDOW_ID] = window_id
                new_attributes[ATTR_WINDOW_HAS_SUN] = window_detail.is_in_sun
                new_attributes[ATTR_SUN_INTENSITY] = round(window_detail.intensity_factor, 3)
                new_attributes[ATTR_SUN_ANGLE] = round(window_detail.sun_angle_to_normal_deg, 1)

                # Update entity state with new attributes
                hass.states.async_set(
                    shade_entity_id,
                    shade_entity.state,
                    new_attributes,
                )
                updated_count += 1

            _LOGGER.debug(
                f"Updated {updated_count} shade entities with sun data "
                f"(azimuth={azimuth:.1f}°, elevation={elevation:.1f}°)"
            )

        except Exception as e:
            _LOGGER.error(f"Error updating shade sun attributes: {e}", exc_info=True)

    # Schedule periodic updates
    async_track_time_interval(
        hass,
        update_shade_sun_attributes,
        timedelta(seconds=update_interval),
    )

    _LOGGER.info(
        f"Sun shade integration initialized. "
        f"Updates every {update_interval} seconds."
    )

    # Run initial update
    await update_shade_sun_attributes()

    return True
