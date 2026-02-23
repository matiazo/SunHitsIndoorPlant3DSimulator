# Sun Hits Indoor Plant 3D Simulator

3D ray-casting simulator that determines if/when direct sunlight reaches an indoor plant through windows. Includes a Home Assistant custom integration for real-time shade control.

## Project Structure

- `sun_hit_detector/` — Core Python package: models, coordinate math, window sun checks, ray casting
- `custom_components/sun_shade_integration/` — HA custom integration (config flow v2, sensor entities)
- `config/` — Default simulation config JSON
- `examples/` — Standalone scripts for testing and yearly analysis
- `scripts/` — Utility scripts (e.g. fetching HA shade entities)

## Key Documentation

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) — How to deploy the HA integration to a server (docker, volume mounts, scp)
- [HOMEASSISTANT_SETUP.md](HOMEASSISTANT_SETUP.md) — HA integration setup instructions and architecture
- [WINDOW_SUN_IMPLEMENTATION_PLAN.md](WINDOW_SUN_IMPLEMENTATION_PLAN.md) — Design doc for the window sun exposure feature
- [YEARLY_SIDE_WINDOW_ANALYSIS.md](YEARLY_SIDE_WINDOW_ANALYSIS.md) — Analysis of yearly sun patterns through side windows
- [custom_components/sun_shade_integration/README.md](custom_components/sun_shade_integration/README.md) — HA integration docs: config flow wizard, entity reference, troubleshooting

## Key Files

| File | Purpose |
|---|---|
| `sun_hit_detector/core/models.py` | `Config`, `Wall`, `Window`, `Plant`, `WindowSunResult` dataclasses; `Config.from_dict()` parses geometry |
| `sun_hit_detector/core/window_sun.py` | `check_windows_from_config()` — main sun exposure calculation |
| `sun_hit_detector/core/coordinates.py` | Coordinate transforms, `position_from_wall_distances()` |
| `custom_components/sun_shade_integration/__init__.py` | HA entry setup, `DataUpdateCoordinator`, `_build_config_dict()` |
| `custom_components/sun_shade_integration/config_flow.py` | 4-step config wizard + menu-based options flow |

## Coordinate System

- Two walls meet at corner origin (0, 0)
- Wall axis `x`: wall runs along X-axis, windows positioned via `position_along_wall` (X distance from corner)
- Wall axis `y`: wall runs along Y-axis, windows positioned via `position_along_wall` (Y distance from corner)
- Plant uses `dist_from_wall1` / `dist_from_wall2` (perpendicular distances into room)
- `outward_normal_azimuth_deg`: compass bearing of wall's outward normal (0=N, 90=E, 180=S, 270=W)

## Deployment

Target: `master@kube1.local` — HA runs in Docker with `/sun-hit-detector` volume mount. See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

## Development

- Python 3.11+, numpy required
- No test framework configured yet
- HA integration config flow VERSION = 2 (v1 JSON-based entries are rejected)
