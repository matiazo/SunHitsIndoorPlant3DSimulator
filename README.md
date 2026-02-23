# Sun Shade Integration for Home Assistant

A Home Assistant custom integration that uses 3D ray-casting simulation to determine when direct sunlight reaches indoor plants through windows.

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
                +-----+  \          |  Home Assistant Entities      |
                       \  \         |------------------------------|
                    /---\--\        |  window_1a has sun:  ON       |
       ROOM        | //  \ |       |  window_1a intensity: 78%     |
                   | PLANT |       |  window_1a angle:    25 deg   |
                   | \\  / |       |                               |
                   +-------+       |  Plant sun start:  13:45      |
                                   |  Plant sun end:    15:30      |
                                   |  Plant sun duration: 120 min  |
                                   +-------------------------------+
```

### Entities Created

**Per window (real-time, updated every N seconds):**
- `binary_sensor.<window>_has_sun` — whether the window is receiving direct sunlight
- `sensor.<window>_sun_intensity` — intensity factor (0-100%)
- `sensor.<window>_sun_angle` — angle between sun and window normal (0-90 deg)

**Plant-level (daily forecast, computed once per day):**
- `sensor.plant_sun_start` — first time sun hits the plant today (like sunrise)
- `sensor.plant_sun_end` — last time sun hits the plant today (like sunset)
- `sensor.plant_sun_duration` — total sun exposure in minutes (like day length)

## Installation

### HACS (Recommended)

1. Open HACS in your Home Assistant instance
2. Click the three dots in the top right corner and select **Custom repositories**
3. Add `https://github.com/matiazo/SunHitsIndoorPlant3DSimulator` with category **Integration**
4. Click **Install**
5. Restart Home Assistant
6. Go to **Settings > Integrations > Add Integration** and search for "Sun Shade Integration"

### Manual

1. Copy the `custom_components/sun_shade_integration` folder into your Home Assistant `config/custom_components/` directory
2. Restart Home Assistant
3. Go to **Settings > Integrations > Add Integration** and search for "Sun Shade Integration"

## Documentation

See the full integration docs at [`custom_components/sun_shade_integration/README.md`](custom_components/sun_shade_integration/README.md) for config flow details, entity reference, and troubleshooting.
