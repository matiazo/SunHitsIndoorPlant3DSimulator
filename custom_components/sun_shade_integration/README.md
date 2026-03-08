# Sun Shade Integration

Custom Home Assistant integration that monitors sun position and exposes real-time window sun exposure data and daily plant sun forecast via 3D ray-casting simulation.

```
                        * Sun
                       /
                      /  azimuth + elevation
                     /
   =================/=================
   ||    WALL    |     |    WALL    ||
   ||            |     |           ||          OUTSIDE
   ||            |  W  |           ||
   ||            |  I  |           ||
   ||            |  N  | ray       ||
   ||            |  D  |/          ||
   =============|  O  |============/
                |  W  |\
                |     | \            +------------------------------+
                +-----+  \          |  Window 1a (Device)           |
                       \  \         |------------------------------|
                    /---\--\        |  has sun:       ON            |
       ROOM        | //  \ |       |  intensity:     78%           |
                   | PLANT |       |  angle:         25 deg        |
                   | \\  / |       |  first light:   13:45         |
                   +-------+       |  last light:    15:30         |
                                   +-------------------------------+
                                   |  Plant (Device)               |
                                   |  sun start:  13:45            |
                                   |  sun end:    16:00            |
                                   |  sun duration: 120 min        |
                                   +-------------------------------+
```

## What It Does

The integration periodically reads the sun's azimuth and elevation from Home Assistant's built-in `sun.sun` entity, runs a 3D ray-casting simulation against your room/window geometry, and exposes results as sensor entities grouped under devices.

### Devices & Entities

**Per-window device** (e.g. "Window window_1a"):

| Entity | Type | Description |
|---|---|---|
| `binary_sensor.<window>_has_sun` | Binary sensor | Whether sun through this window hits the plant |
| `sensor.<window>_sun_intensity` | Sensor (%) | Relative sun intensity factor based on angle of incidence |
| `sensor.<window>_sun_angle` | Sensor (deg) | Angle between the sun direction and the window normal |
| `sensor.<window>_first_light` | Timestamp | First time sun will hit plant through this window today |
| `sensor.<window>_last_light` | Timestamp | Last time sun will hit plant through this window today |

**Plant device** ("Sun Shade Plant"):

| Entity | Type | Description |
|---|---|---|
| `sensor.plant_sun_start` | Timestamp | First time sun hits the plant today (any window) |
| `sensor.plant_sun_end` | Timestamp | Last time sun hits the plant today (any window) |
| `sensor.plant_sun_duration` | Duration (min) | Total sun exposure on the plant today |

All timestamp sensors display natively in HA dashboards (Lovelace cards, entity rows, etc.).

The plant and per-window forecasts are computed once per day by scanning 5am–9pm at 15-minute intervals using the full 3D ray-casting engine. They recalculate automatically when the date changes.

## Prerequisites

- Home Assistant with the `sun` integration enabled (included by default)
- The `sun_hit_detector` Python package mounted inside the HA container

### Docker Compose Volume Mount

```yaml
home-assistant:
  image: ghcr.io/home-assistant/home-assistant:2025.9.3
  volumes:
    - /home/master/homeassistant:/config
    - /home/master/sun-plant-simulator:/sun-plant-simulator:ro
```

## Installation

Copy the `sun_shade_integration` folder into your Home Assistant `custom_components` directory:

```
custom_components/
  sun_shade_integration/
    __init__.py
    binary_sensor.py
    config_flow.py
    const.py
    manifest.json
    sensor.py
    strings.json
```

Then restart Home Assistant.

## Setup via UI

1. Go to **Settings > Devices & Services > + Add Integration**
2. Search for **"Sun Shade Integration"**

You'll see a menu with two options:

### Option A: Import from JSON File (recommended for initial setup)

Provide the path to an existing JSON configuration file (e.g. `/sun-plant-simulator/config/default_config.json`). The integration will parse all walls, windows (with shade entity mappings), and plant data automatically. You can then tweak individual values via the options flow.

### Option B: Configure Manually

The setup wizard guides you through 4 steps:

#### Step 1: General Settings

| Field | Default | Description |
|---|---|---|
| Update interval | `300` (5 min) | How often to recalculate sun exposure (30–3600 seconds) |

#### Step 2: Define Walls (loops)

| Field | Default | Description |
|---|---|---|
| Wall ID | `wall_1`, `wall_2`, ... | Unique identifier for the wall |
| Outward normal azimuth | — | Compass bearing of the wall's outward-facing normal (0=N, 90=E, 180=S, 270=W) |
| Wall thickness | `0.25` | Thickness in meters |
| Wall axis | `x` | Which coordinate axis the wall is aligned with (`x` or `y`) |
| Window count | `1` | Number of windows on this wall |
| Default window dimensions | — | Default width, height, z_bottom, z_top for windows on this wall |
| Add another wall | unchecked | Check to define additional walls |

#### Step 3: Define Windows (loops per wall)

| Field | Default | Description |
|---|---|---|
| Window ID | auto-suggested | Unique identifier (e.g. `window_1a`) |
| Position along wall | — | Distance from wall corner to window's left edge |
| Width/Height/Z bottom/Z top | from wall defaults | Pre-filled from Step 2, editable per window |
| Shade entity | — | Optional: cover entity for this window's shade |

#### Step 4: Plant Position

| Field | Default | Description |
|---|---|---|
| Distance from wall 1 | — | Perpendicular distance from wall 1 to the plant center |
| Distance from wall 2 | — | Perpendicular distance from wall 2 to the plant center |
| Plant radius | `0.3` | Radius of the plant canopy |
| Plant Z min / Z max | `0.0` / `1.2` | Bottom and top elevation of the plant |

## Reconfiguration

After initial setup, you can change settings at any time:

1. Go to **Settings > Devices & Services**
2. Find **Sun Shade Integration** and click **Configure**
3. A menu lets you edit General settings, Walls, Windows, or Plant position individually

## Example Automation: Open Shades When Sun Hits Plant

```yaml
automation:
  - alias: "Open shade when sun hits plant through window"
    trigger:
      - platform: state
        entity_id: binary_sensor.window_1a_has_sun
        to: "on"
    action:
      - service: cover.open_cover
        target:
          entity_id: cover.living_room_front_shade_4

  - alias: "Close shade when sun stops hitting plant through window"
    trigger:
      - platform: state
        entity_id: binary_sensor.window_1a_has_sun
        to: "off"
    action:
      - service: cover.close_cover
        target:
          entity_id: cover.living_room_front_shade_4
```

Repeat for each window/shade pair. The binary sensor is ON only when sun through that specific window reaches the plant — so only the required shades open.

## How It Works

```
sun.sun entity (azimuth, elevation)             hass.config (lat, lon, tz)
        |                                               |
        v                                               v
check_windows_from_config()              generate_sun_data_for_date()
  3D ray-casting per window                scan 5am-9pm @ 15-min intervals
        |                                               |
        v                                               v
check_plant_hit_per_window()             check_plant_hit_per_window()
  per-window plant hit test                per-window hit at each interval
        |                                               |
        v                                               v
+---------------------------+            +-------------------------------+
|  Per-window sensors       |            |  Per-window + plant forecast  |
|  (every N seconds)        |            |  (once per day, cached)       |
|                           |            |                               |
|  has_sun (binary)         |            |  window first/last light      |
|  intensity                |            |  plant sun start/end/duration |
|  angle                    |            +-------------------------------+
+---------------------------+
```

## File Reference

| File | Purpose |
|---|---|
| `__init__.py` | Integration setup/teardown, `_build_config_dict()`, coordinator with real-time + daily forecast |
| `sensor.py` | Window intensity/angle/first light/last light sensors + plant sun start/end/duration sensors |
| `binary_sensor.py` | Per-window has-sun binary sensors |
| `config_flow.py` | Menu-based config flow (manual wizard or JSON import) and options flow |
| `const.py` | Domain name, config keys, attribute name constants |
| `manifest.json` | Integration metadata, dependencies, config_flow flag |
| `strings.json` | UI labels, descriptions, and error messages for the config flow |

## Troubleshooting

### Integration not found in Add Integration

- Verify the files are in `/config/custom_components/sun_shade_integration/`
- Check that `manifest.json` has `"config_flow": true`
- Restart Home Assistant after copying files

### "Could not import sun_hit_detector" error

- Verify the volume mount exists: `docker exec home-assistant ls /sun-plant-simulator/sun_hit_detector/`
- Check docker-compose has the volume mount

### Logs

```bash
docker logs home-assistant 2>&1 | grep sun_shade
docker logs -f home-assistant 2>&1 | grep sun_shade
```
