# Deployment Guide

## Installation

See [HOMEASSISTANT_SETUP.md](HOMEASSISTANT_SETUP.md) for full installation and configuration instructions.

**TL;DR**: Install via HACS (recommended) or copy `custom_components/sun_shade_integration/` manually, then configure via the HA UI config flow.

## Window to Shade Mapping

Current configuration (8 windows across 2 walls):

#### Wall 1 (Front) — Azimuth 210°, inverted shade order
| Window ID | Position | Shade Entity ID |
|-----------|----------|-----------------|
| window_1a | x=0.36 | `cover.living_room_front_shade_4` |
| window_1b | x=1.44 | `cover.living_room_front_shade_3` |
| window_1c | x=2.52 | `cover.living_room_front_shade_2` |
| window_1d | x=3.60 | `cover.living_room_front_shade_1` |

#### Wall 2 (Side) — Azimuth 300°, normal order
| Window ID | Position | Shade Entity ID |
|-----------|----------|-----------------|
| window_2a | y=0.80 | `cover.living_room_side_shade_1` |
| window_2b | y=4.00 | `cover.living_room_side_shade_2` |
| window_2c | y=5.07 | `cover.living_room_side_shade_3` |
| window_2d | y=8.26 | `cover.living_room_side_shade_4` |

> **Note**: This mapping is stored in the HA config entry (configured via UI). The `config/default_config.json` in this repo is only used by the standalone CLI tool.

## Local Testing (Standalone CLI)

The `check_plant_sun.py` CLI tool can be used independently of Home Assistant for testing and analysis:

```bash
# Test window sun detection with current sun position
python check_plant_sun.py --config config/default_config.json --windows --json

# Test specific azimuth/elevation
python check_plant_sun.py 210 30 --config config/default_config.json --windows

# Run yearly simulation
python examples/simulate_yearly_plant_sun.py
```

## Example Automations

### Auto-Close Shades When Window Gets Sun

```yaml
automation:
  - alias: "Close front shade 4 when window gets direct sun"
    trigger:
      - platform: state
        entity_id: binary_sensor.window_1a_has_sun
        to: "on"
    condition:
      - condition: numeric_state
        entity_id: sensor.window_1a_sun_intensity
        above: 50
      - condition: time
        after: "10:00:00"
        before: "18:00:00"
    action:
      - service: cover.close_cover
        target:
          entity_id: cover.living_room_front_shade_4
```

### Dashboard Card

```yaml
type: entities
title: Window Sun Exposure
entities:
  - entity: binary_sensor.window_1a_has_sun
  - entity: sensor.window_1a_sun_intensity
  - entity: sensor.plant_sun_start
  - entity: sensor.plant_sun_end
  - entity: sensor.plant_sun_duration
```
