# Sun Shade Integration for Home Assistant

A Home Assistant custom integration that uses 3D ray-casting simulation to determine when direct sunlight reaches indoor plants through windows. Creates real-time sensors for sun exposure per window.

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
