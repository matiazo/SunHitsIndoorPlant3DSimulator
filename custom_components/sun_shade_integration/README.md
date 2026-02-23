# Sun Shade Integration

Custom Home Assistant integration that monitors sun position and updates shade (cover) entity attributes with real-time window sun exposure data.

## What It Does

The integration periodically reads the sun's azimuth and elevation from Home Assistant's built-in `sun.sun` entity, runs a 3D ray-casting simulation against your room/window geometry, and writes the results as extra attributes on your shade entities.

### Attributes Added to Shade Entities

| Attribute | Type | Description |
|---|---|---|
| `window_id` | string | The window ID associated with this shade (e.g. `window_1a`) |
| `window_has_sun` | boolean | Whether the window is receiving direct sunlight |
| `sun_intensity` | float (0.0-1.0) | Relative sun intensity factor based on angle of incidence |
| `sun_angle_deg` | float (0-90) | Angle in degrees between the sun direction and the window normal |

## Prerequisites

- Home Assistant with the `sun` integration enabled (included by default)
- The `sun_hit_detector` Python package mounted inside the HA container at `/sun-hit-detector`

### Docker Compose Volume Mount

The HA container must have the simulator package mounted as a read-only volume:

```yaml
home-assistant:
  image: ghcr.io/home-assistant/home-assistant:2025.9.3
  volumes:
    - /home/master/homeassistant:/config
    - /home/master/sun-hit-detector:/sun-hit-detector:ro
```

## Installation

Copy the `sun_shade_integration` folder into your Home Assistant `custom_components` directory:

```
custom_components/
  sun_shade_integration/
    __init__.py
    config_flow.py
    const.py
    manifest.json
    strings.json
```

Then restart Home Assistant.

## Setup via UI

1. Go to **Settings > Devices & Services > + Add Integration**
2. Search for **"Sun Shade Integration"**

The setup wizard guides you through 4 steps:

### Step 1: General Settings

| Field | Default | Description |
|---|---|---|
| Update interval | `300` (5 min) | How often to recalculate sun exposure (30–3600 seconds) |

### Step 2: Define Walls (loops)

For each wall in your room, enter:

| Field | Default | Description |
|---|---|---|
| Wall ID | `wall_1`, `wall_2`, ... | Unique identifier for the wall |
| Outward normal azimuth | — | Compass bearing of the wall's outward-facing normal (0=N, 90=E, 180=S, 270=W) |
| Wall thickness | `0.25` | Thickness in meters |
| Wall axis | `x` | Which coordinate axis the wall is aligned with (`x` or `y`) |
| Window count | `1` | Number of windows on this wall |
| Default window width | `0.89` | Default width for windows on this wall |
| Default window height | `1.50` | Default height for windows on this wall |
| Default Z bottom | `4.2` | Default bottom elevation of windows |
| Default Z top | `5.7` | Default top elevation of windows |
| Add another wall | unchecked | Check to define additional walls |

### Step 3: Define Windows (loops per wall)

For each window (auto-loops based on window count per wall):

| Field | Default | Description |
|---|---|---|
| Window ID | auto-suggested | Unique identifier (e.g. `window_1a`) |
| Position along wall | — | Distance from wall corner to window's left edge |
| Width/Height/Z bottom/Z top | from wall defaults | Pre-filled from Step 2, editable per window |
| Shade entity | — | Optional: cover entity for this window |

### Step 4: Plant Position

| Field | Default | Description |
|---|---|---|
| Distance from wall 1 | — | Perpendicular distance from wall 1 to the plant center |
| Distance from wall 2 | — | Perpendicular distance from wall 2 to the plant center |
| Plant radius | `0.3` | Radius of the plant canopy |
| Plant Z min | `0.0` | Bottom elevation of the plant |
| Plant Z max | `1.2` | Top elevation of the plant |

## Reconfiguration

After initial setup, you can change settings at any time:

1. Go to **Settings > Devices & Services**
2. Find **Sun Shade Integration** and click **Configure**
3. The options flow re-runs the full wizard, pre-populated with your current values

## Config Entry Data

The config entry stores all geometry in a structured dict:

```json
{
  "update_interval": 300,
  "walls": [
    {"id": "wall_1", "outward_normal_azimuth_deg": 210.0, "thickness": 0.25, "axis": "x"},
    {"id": "wall_2", "outward_normal_azimuth_deg": 300.0, "thickness": 0.25, "axis": "y"}
  ],
  "windows": [
    {"id": "window_1a", "wall_id": "wall_1", "position_along_wall": 0.36,
     "width": 0.89, "height": 1.50, "z_bottom": 4.2, "z_top": 5.7,
     "shade_entity_id": "cover.living_room_front_shade_4"}
  ],
  "plant": {"dist_from_wall1": 8.0, "dist_from_wall2": 3.9, "radius": 0.3, "z_min": 0.0, "z_max": 1.2}
}
```

At runtime, `_build_config_dict()` wraps this into the format `Config.from_dict()` expects (adds `coordinate_system`, `corner`, `simulation` defaults).

## Migration from v1

If you previously used the JSON-file-based configuration (config flow VERSION 1), you must remove the integration and re-add it. The new UI wizard replaces the JSON file entirely.

## How It Works

```
sun.sun entity (azimuth, elevation)
        |
        v
check_windows_from_config()    <-- sun_hit_detector 3D ray-casting
        |
        v
For each window with a mapped shade:
  - Read current shade entity state
  - Merge in window_has_sun, sun_intensity, sun_angle_deg attributes
  - Write updated state back
```

The update runs once immediately on startup and then at the configured interval.

## File Reference

| File | Purpose |
|---|---|
| `__init__.py` | Integration setup/teardown, `_build_config_dict()`, periodic sun calculation update loop |
| `config_flow.py` | 4-step UI config flow (walls/windows/plant wizard) and options flow |
| `const.py` | Domain name, config keys, attribute name constants |
| `manifest.json` | Integration metadata, dependencies, config_flow flag |
| `strings.json` | UI labels, descriptions, and error messages for the config flow |

## Troubleshooting

### Integration not found in Add Integration

- Verify the files are in `/config/custom_components/sun_shade_integration/`
- Check that `manifest.json` has `"config_flow": true`
- Restart Home Assistant after copying files

### "Could not import sun_hit_detector" error

- Verify the volume mount exists: `docker exec home-assistant ls /sun-hit-detector/sun_hit_detector/`
- Check docker-compose has `/home/master/sun-hit-detector:/sun-hit-detector:ro`

### Attributes not appearing on shade entities

- Check that the `sun.sun` entity is available (Developer Tools > States)
- Verify shade entity IDs were selected in the window configuration steps
- Check logs: `docker logs home-assistant 2>&1 | grep sun_shade`

### Logs

```bash
# All component logs
docker logs home-assistant 2>&1 | grep sun_shade

# Live follow
docker logs -f home-assistant 2>&1 | grep sun_shade
```

## Deployment

```bash
# 1. Copy simulator package
scp -r sun_hit_detector master@<host>:/home/master/sun-hit-detector-pkg
ssh master@<host> "mkdir -p /home/master/sun-hit-detector && mv /home/master/sun-hit-detector-pkg /home/master/sun-hit-detector/sun_hit_detector"

# 2. Copy custom component (via docker cp if custom_components is root-owned)
scp -r custom_components/sun_shade_integration master@<host>:/tmp/sun_shade_integration
ssh master@<host> "docker cp /tmp/sun_shade_integration home-assistant:/config/custom_components/sun_shade_integration"

# 3. Add volume mount to docker-compose.yml (under home-assistant service volumes)
#    - /home/master/sun-hit-detector:/sun-hit-detector:ro

# 4. Recreate container and start
ssh master@<host> "docker rm home-assistant && cd /home/master && docker-compose up -d home-assistant"
```
