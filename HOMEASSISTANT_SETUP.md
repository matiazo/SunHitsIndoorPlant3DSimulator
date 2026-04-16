# Home Assistant Integration Setup

## Prerequisites

- Home Assistant **2024.1.0+** (Docker or HA OS)
- Python package `sun-hit-detector` installed in the HA environment

## Installation

### Option A: HACS (Recommended)

1. Install [HACS](https://hacs.xyz/) if not already installed
2. Add this repository as a **custom repository** in HACS:
   - URL: `https://github.com/matiazo/SunHitsIndoorPlant3DSimulator`
   - Category: **Integration**
3. Search for "Sun Shade Integration" in HACS and install
4. Restart Home Assistant

### Option B: Manual Install

1. Copy `custom_components/sun_shade_integration/` into your HA config directory:
   ```
   <ha-config>/custom_components/sun_shade_integration/
   ```
2. Install the core library inside the HA container:
   ```bash
   docker exec home-assistant pip install sun-hit-detector
   ```
3. Restart Home Assistant

## Configuration (UI Config Flow)

All configuration is done via the Home Assistant UI — **no YAML configuration needed**.

1. Go to **Settings → Devices & Services → Add Integration**
2. Search for **"Sun Shade Integration"**
3. Follow the config flow wizard:
   - **Step 1**: Choose manual config or import from JSON file
   - **Step 2**: Define walls (axis, outward normal azimuth, thickness)
   - **Step 3**: Define windows per wall (position, size, optional shade entity)
   - **Step 4**: Define plant position (perpendicular distances from each wall)
4. The integration creates entities automatically

## Entities Created

### Per Window
| Entity | Type | Description |
|--------|------|-------------|
| `binary_sensor.<window_id>_has_sun` | Binary Sensor | Whether window receives direct sun |
| `sensor.<window_id>_sun_intensity` | Sensor (%) | Sun intensity factor (0–100%) |
| `sensor.<window_id>_sun_angle` | Sensor (°) | Angle between sun and window normal |
| `sensor.<window_id>_first_light` | Sensor (timestamp) | Forecast: first time sun hits plant through this window today |
| `sensor.<window_id>_last_light` | Sensor (timestamp) | Forecast: last time sun hits plant through this window today |

### Plant Level
| Entity | Type | Description |
|--------|------|-------------|
| `sensor.plant_sun_start` | Sensor (timestamp) | First time sun hits the plant today |
| `sensor.plant_sun_end` | Sensor (timestamp) | Last time sun hits the plant today |
| `sensor.plant_sun_duration` | Sensor (minutes) | Total sun exposure duration today |

## Reconfiguring

To edit walls, windows, or plant position after setup:

1. Go to **Settings → Devices & Services**
2. Find **Sun Shade Integration** and click **Configure**
3. Use the menu to edit general settings, walls, windows, or plant position

## Updating the Integration

### Via HACS
HACS will notify you of updates. Click **Update** and restart HA.

### Manual Update
1. Download the latest `custom_components/sun_shade_integration/` from this repository
2. Replace the files in `<ha-config>/custom_components/sun_shade_integration/`
3. Restart Home Assistant

## Troubleshooting

### Integration Not Loading
Check HA logs for errors:
```bash
docker logs home-assistant 2>&1 | grep sun_shade
```

Common causes:
- Missing `sun-hit-detector` pip package — install with `pip install sun-hit-detector`
- Old v1 config entry — remove and re-add the integration via UI

### Sensor Not Updating
- Default update interval is 300 seconds (5 minutes), configurable via options flow
- Verify `sun.sun` entity is available in Developer Tools → States

### Migration from v1
If you previously used the JSON file-based configuration (v1), you must:
1. Remove the old integration entry from **Settings → Devices & Services**
2. Delete any leftover files: `sun_plant_config.json`, `sun_plant_sensor.yaml`
3. Remove any `command_line:` or `sun_shade_integration:` entries from `configuration.yaml`
4. Remove any `/sun-hit-detector` volume mounts from your `docker-compose.yml`
5. Re-add the integration via the UI config flow
