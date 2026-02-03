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


# Window sun exposure functions


def get_window_sun_status(
    sun_azimuth: float,
    sun_elevation: float,
    config_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Get sun exposure status for all windows.

    This function checks which windows are receiving direct sunlight and returns
    detailed information for each window including intensity and sun angle.

    Args:
        sun_azimuth: Current sun azimuth in degrees (from HA sun.sun entity).
        sun_elevation: Current sun elevation in degrees (from HA sun.sun entity).
        config_path: Optional path to config file.

    Returns:
        Dictionary with:
        - windows_in_sun: List of window IDs currently receiving sun
        - window_details: Dict mapping window_id to details (is_in_sun, angle, intensity)
        - sun_azimuth_deg: The input azimuth
        - sun_elevation_deg: The input elevation
        - reason: Explanation if no windows in sun (e.g., "sun_below_horizon")

    Example:
        >>> status = get_window_sun_status(210, 30)
        >>> print(f"Windows in sun: {status['windows_in_sun']}")
        >>> for wid, details in status['window_details'].items():
        ...     print(f"{wid}: {details['intensity_factor']:.2f}")
    """
    from ..core.window_sun import check_windows_from_config

    config = load_config(config_path)
    result = check_windows_from_config(
        sun_azimuth_deg=sun_azimuth,
        sun_elevation_deg=sun_elevation,
        config=config,
        use_ray_validation=True,
    )

    return result.to_dict()


def check_window_sunlight(
    window_id: str,
    sun_azimuth: float,
    sun_elevation: float,
    config_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Check if a specific window is currently receiving direct sunlight.

    Simple boolean function for individual window checks. Useful for HA
    command-line integrations and simple automations.

    Args:
        window_id: ID of the window to check (e.g., "window_1a").
        sun_azimuth: Current sun azimuth in degrees.
        sun_elevation: Current sun elevation in degrees.
        config_path: Optional path to config file.

    Returns:
        True if the window is receiving direct sunlight, False otherwise.

    Example:
        >>> is_sunny = check_window_sunlight("window_1a", 210, 30)
        >>> if is_sunny:
        ...     print("Window 1A has sun!")
    """
    status = get_window_sun_status(sun_azimuth, sun_elevation, config_path)
    return window_id in status["windows_in_sun"]


def get_window_intensity(
    window_id: str,
    sun_azimuth: float,
    sun_elevation: float,
    config_path: Optional[Union[str, Path]] = None,
) -> float:
    """Get the sun intensity factor for a specific window.

    Returns a value from 0.0 to 1.0 representing the relative intensity of
    sunlight at the window. 1.0 means sun is directly perpendicular to window,
    lower values mean sun is at an oblique angle. 0.0 means no direct sun.

    Useful for advanced automations like HVAC load calculations or shade
    positioning based on intensity.

    Args:
        window_id: ID of the window to check (e.g., "window_1a").
        sun_azimuth: Current sun azimuth in degrees.
        sun_elevation: Current sun elevation in degrees.
        config_path: Optional path to config file.

    Returns:
        Intensity factor from 0.0 to 1.0. Returns 0.0 if window not in sun.

    Example:
        >>> intensity = get_window_intensity("window_1a", 210, 30)
        >>> if intensity > 0.5:
        ...     print("Strong sunlight, close shades!")
    """
    status = get_window_sun_status(sun_azimuth, sun_elevation, config_path)
    window_detail = status["window_details"].get(window_id)

    if window_detail is None:
        return 0.0

    return window_detail.get("intensity_factor", 0.0)


def get_shade_sun_info(
    shade_entity_id: str,
    sun_azimuth: float,
    sun_elevation: float,
    config_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Get sun exposure information by shade entity ID.

    Query by shade entity ID instead of window ID. This is useful when you
    want to check "Does this shade's window have sun?" without knowing the
    underlying window ID.

    Args:
        shade_entity_id: HA entity ID of the shade (e.g., "cover.living_room_shade_1a").
        sun_azimuth: Current sun azimuth in degrees.
        sun_elevation: Current sun elevation in degrees.
        config_path: Optional path to config file.

    Returns:
        Dictionary with:
        - window_id: The window ID associated with this shade (or None)
        - is_in_sun: Boolean indicating if window is receiving sun
        - intensity: Intensity factor (0.0-1.0)
        - angle: Sun angle to window normal in degrees

    Example:
        >>> info = get_shade_sun_info("cover.living_room_shade_1a", 210, 30)
        >>> if info['is_in_sun'] and info['intensity'] > 0.5:
        ...     print("Close the shade!")
    """
    config = load_config(config_path)

    # Find window with matching shade_entity_id
    matching_window = None
    for window in config.windows:
        if window.shade_entity_id == shade_entity_id:
            matching_window = window
            break

    if matching_window is None:
        return {
            "window_id": None,
            "is_in_sun": False,
            "intensity": 0.0,
            "angle": 90.0,
            "error": f"No window found with shade_entity_id '{shade_entity_id}'",
        }

    # Get window sun status
    status = get_window_sun_status(sun_azimuth, sun_elevation, config_path)
    window_detail = status["window_details"].get(matching_window.id)

    if window_detail is None:
        return {
            "window_id": matching_window.id,
            "is_in_sun": False,
            "intensity": 0.0,
            "angle": 90.0,
        }

    return {
        "window_id": matching_window.id,
        "is_in_sun": window_detail["is_in_sun"],
        "intensity": window_detail["intensity_factor"],
        "angle": window_detail["sun_angle_to_normal_deg"],
    }
