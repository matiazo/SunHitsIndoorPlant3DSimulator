"""Constants for sun_shade_integration."""

DOMAIN = "sun_shade_integration"
CONF_CONFIG_PATH = "config_path"
CONF_UPDATE_INTERVAL = "update_interval"
DEFAULT_UPDATE_INTERVAL = 300  # 5 minutes

# Attribute names added to shade entities
ATTR_WINDOW_ID = "window_id"
ATTR_WINDOW_HAS_SUN = "window_has_sun"
ATTR_SUN_INTENSITY = "sun_intensity"
ATTR_SUN_ANGLE = "sun_angle_deg"
