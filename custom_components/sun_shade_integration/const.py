"""Constants for sun_shade_integration."""

DOMAIN = "sun_shade_integration"

# Top-level config entry keys
CONF_UPDATE_INTERVAL = "update_interval"
CONF_WALLS = "walls"
CONF_WINDOWS = "windows"
CONF_PLANT = "plant"

DEFAULT_UPDATE_INTERVAL = 300  # 5 minutes

# Wall field keys
CONF_WALL_ID = "wall_id"
CONF_OUTWARD_NORMAL = "outward_normal_azimuth_deg"
CONF_WALL_THICKNESS = "wall_thickness"
CONF_WALL_AXIS = "wall_axis"
CONF_WINDOW_COUNT = "window_count"
CONF_DEFAULT_WINDOW_WIDTH = "default_window_width"
CONF_DEFAULT_WINDOW_HEIGHT = "default_window_height"
CONF_DEFAULT_Z_BOTTOM = "default_z_bottom"
CONF_DEFAULT_Z_TOP = "default_z_top"
CONF_ADD_ANOTHER_WALL = "add_another_wall"

# Window field keys
CONF_WINDOW_ID = "window_id"
CONF_POSITION_ALONG_WALL = "position_along_wall"
CONF_WINDOW_WIDTH = "window_width"
CONF_WINDOW_HEIGHT = "window_height"
CONF_Z_BOTTOM = "z_bottom"
CONF_Z_TOP = "z_top"
CONF_SHADE_ENTITY_ID = "shade_entity_id"

# Plant field keys
CONF_PLANT_DIST_WALL1 = "plant_dist_wall1"
CONF_PLANT_DIST_WALL2 = "plant_dist_wall2"
CONF_PLANT_RADIUS = "plant_radius"
CONF_PLANT_Z_MIN = "plant_z_min"
CONF_PLANT_Z_MAX = "plant_z_max"

# Config file import keys
CONF_CONFIG_FILE_PATH = "config_file_path"
DEFAULT_CONFIG_FILE_PATH = "/sun-plant-simulator/config/default_config.json"

# Attribute names added to shade entities
ATTR_WINDOW_ID = "window_id"
ATTR_WINDOW_HAS_SUN = "window_has_sun"
ATTR_SUN_INTENSITY = "sun_intensity"
ATTR_SUN_ANGLE = "sun_angle_deg"
