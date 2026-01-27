"""Home Assistant integration service functions.

This module provides simple, automation-friendly functions for use with
Home Assistant. These functions are designed to be called from HA Python
scripts or custom components.

Example Home Assistant automation:
    ```yaml
    automation:
      - alias: "Notify when plant gets sun"
        trigger:
          platform: state
          entity_id: sun.sun
        action:
          - service: python_script.check_plant_sunlight
    ```

Example Python script (python_scripts/check_plant_sunlight.py):
    ```python
    from sun_plant_simulator.homeassistant import check_sunlight

    sun = hass.states.get("sun.sun")
    azimuth = float(sun.attributes.get("azimuth", 0))
    elevation = float(sun.attributes.get("elevation", 0))

    is_hit = check_sunlight(azimuth, elevation)

    if is_hit:
        hass.services.call("notify", "mobile_app", {
            "message": "Your plant is getting direct sunlight!"
        })
    ```
"""

from pathlib import Path
from typing import Optional, Union

from ..core.hit_test import check_sun_hits_plant
from ..core.models import Config

# Default config path (can be overridden)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default_config.json"

# Cached config to avoid reloading on every call
_cached_config: Optional[Config] = None
_cached_config_path: Optional[str] = None


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration, with caching for performance.

    Args:
        config_path: Path to config file. If None, uses default.

    Returns:
        Config object.
    """
    global _cached_config, _cached_config_path

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path_str = str(config_path)

    # Return cached config if path matches
    if _cached_config is not None and _cached_config_path == config_path_str:
        return _cached_config

    # Load and cache
    _cached_config = Config.from_json_file(config_path)
    _cached_config_path = config_path_str

    return _cached_config


def clear_config_cache() -> None:
    """Clear the cached configuration.

    Call this if you've modified the config file and want to reload it.
    """
    global _cached_config, _cached_config_path
    _cached_config = None
    _cached_config_path = None


def check_sunlight(
    sun_azimuth: float,
    sun_elevation: float,
    config_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Check if the plant is currently receiving direct sunlight.

    This is the simplest function for Home Assistant automations.
    Returns a single boolean that can be used directly in conditions.

    Args:
        sun_azimuth: Current sun azimuth in degrees (from HA sun.sun entity).
        sun_elevation: Current sun elevation in degrees (from HA sun.sun entity).
        config_path: Optional path to config file.

    Returns:
        True if plant is receiving direct sunlight through a window.

    Example:
        >>> from sun_plant_simulator.homeassistant import check_sunlight
        >>> is_sunny = check_sunlight(azimuth=180, elevation=45)
        >>> if is_sunny:
        ...     print("Move the blinds!")
    """
    config = load_config(config_path)
    result = check_sun_hits_plant(
        sun_azimuth_deg=sun_azimuth,
        sun_elevation_deg=sun_elevation,
        plant=config.plant,
        windows=config.windows,
        n_angular=config.simulation.sample_points_angular,
        n_vertical=config.simulation.sample_points_vertical,
    )
    return result.is_hit


def get_sunlight_details(
    sun_azimuth: float,
    sun_elevation: float,
    config_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Get detailed information about current sunlight status.

    This function returns more information than check_sunlight(),
    useful for displaying in HA dashboards or for debugging.

    Args:
        sun_azimuth: Current sun azimuth in degrees.
        sun_elevation: Current sun elevation in degrees.
        config_path: Optional path to config file.

    Returns:
        Dictionary with:
        - is_hit: Boolean indicating if plant receives direct sun
        - window_id: ID of window through which light passes (if hit)
        - reason: Explanation if not hit
        - sun_azimuth: The input azimuth
        - sun_elevation: The input elevation
        - n_hit_points: Number of sample points receiving light

    Example:
        >>> details = get_sunlight_details(180, 45)
        >>> print(f"Hit: {details['is_hit']}, Window: {details['window_id']}")
    """
    config = load_config(config_path)
    result = check_sun_hits_plant(
        sun_azimuth_deg=sun_azimuth,
        sun_elevation_deg=sun_elevation,
        plant=config.plant,
        windows=config.windows,
        n_angular=config.simulation.sample_points_angular,
        n_vertical=config.simulation.sample_points_vertical,
    )

    return {
        "is_hit": result.is_hit,
        "window_id": result.window_id,
        "reason": result.reason,
        "sun_azimuth": sun_azimuth,
        "sun_elevation": sun_elevation,
        "n_hit_points": len(result.hit_points),
    }


def get_sunlight_state(
    sun_azimuth: float,
    sun_elevation: float,
    config_path: Optional[Union[str, Path]] = None,
) -> str:
    """Get a human-readable state string for the current sunlight status.

    Useful for template sensors in Home Assistant.

    Args:
        sun_azimuth: Current sun azimuth in degrees.
        sun_elevation: Current sun elevation in degrees.
        config_path: Optional path to config file.

    Returns:
        One of: "direct_sun", "below_horizon", "no_window_path"

    Example HA template sensor:
        ```yaml
        sensor:
          - platform: template
            sensors:
              plant_sunlight:
                value_template: >
                  {% set result = state_attr('sensor.sun_plant', 'state') %}
                  {{ result }}
        ```
    """
    config = load_config(config_path)
    result = check_sun_hits_plant(
        sun_azimuth_deg=sun_azimuth,
        sun_elevation_deg=sun_elevation,
        plant=config.plant,
        windows=config.windows,
        n_angular=config.simulation.sample_points_angular,
        n_vertical=config.simulation.sample_points_vertical,
    )

    if result.is_hit:
        return "direct_sun"
    elif result.reason == "sun_below_horizon":
        return "below_horizon"
    else:
        return "no_window_path"
