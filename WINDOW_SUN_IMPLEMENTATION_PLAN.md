# Implementation Plan: Window Sun Exposure Tracking

## Overview

Add window sun exposure tracking to the SunHitsIndoorPlant3DSimulator. This feature will determine which windows are receiving direct sunlight at any given time, independent of plant position, and integrate with existing Home Assistant shade entities by adding window sun attributes.

**User Requirements:**
- Use ray-based validation for accuracy
- Integrate with existing shade entities via template sensors
- Track intensity factor (0-1) in addition to binary status
- Add shade entity mapping to config file (window-to-shade associations)

## Architecture

The solution uses a **two-stage approach** for sun detection:

1. **Geometric pre-filter**: Quick dot product check (sun direction · window normal) to filter windows that face away from sun
2. **Ray-based validation**: For windows passing the geometric check, cast rays from window center toward sun to verify no obstructions

This provides accuracy (catches obstructions) while maintaining performance (geometric filter eliminates most non-viable windows instantly).

### Integration Strategy

**Custom Home Assistant Component** that directly extends existing shade entities with window sun attributes.

**Benefits:**
- ✅ No duplicate sensors - attributes added directly to existing shade entities
- ✅ Natural integration: `cover.living_room_shade_1a` has `window_has_sun` attribute
- ✅ Automatic updates when sun position changes
- ✅ Clean UX: single entity for both shade control and sun status
- ✅ Future-proof: easy to add more attributes (solar heat gain, UV index, etc.)

**Architecture Flow:**
```
HA sun.sun entity (azimuth, elevation change)
    ↓ (state change event)
Custom Component Listener
    ↓ (extract azimuth, elevation)
Call check_windows_from_config() [our Python code]
    ↓ (returns WindowSunResult with all window data)
Component updates shade entity attributes
    ↓ (hass.states.async_set with extra attributes)
Existing shade entities now have:
  - window_id
  - window_has_sun (bool)
  - sun_intensity (0-1)
  - sun_angle_deg (0-90)
```

**How it works:**
1. Custom component loads on HA startup
2. Reads config file to map windows → shade entities
3. Subscribes to `sun.sun` state changes (or updates every 5 minutes)
4. When sun position changes:
   - Extracts azimuth/elevation from `state_attr('sun.sun', 'azimuth')`
   - Calls our ray calculation: `check_windows_from_config(azimuth, elevation, config)`
   - Updates each shade entity with window sun attributes using `hass.states.async_set()`
5. Automations use: `state_attr('cover.living_room_shade_1a', 'window_has_sun')`

## Implementation Steps

### Step 1: Update Config Schema

**File:** `config\default_config.json` (MODIFY)

Add optional `shade_entity_id` field to each window definition:

```json
{
  "id": "window_1a",
  "wall_id": "wall_1",
  "x_position": 0.36,
  "width": 0.89,
  "height": 1.5,
  "z_bottom": 4.2,
  "z_top": 5.7,
  "shade_entity_id": "cover.living_room_shade_1a"
}
```

Repeat for all 8 windows, mapping each to its corresponding shade entity.

**Note:** The shade_entity_id field is optional - windows without shades can omit this field.

### Step 2: Update Config Parser

**File:** `sun_hit_detector\core\models.py` (MODIFY)

1. Add `shade_entity_id` field to Window class (after line 50):
```python
shade_entity_id: Optional[str] = None
```

2. Update `Config.from_dict()` to read shade_entity_id from window definitions (around line 234):
```python
windows.append(
    Window(
        id=w["id"],
        center=np.array(w["center"], dtype=float),
        width=w["width"],
        height=w["height"],
        wall_normal_azimuth=wall_normal,
        wall_id=wall_id,
        wall_thickness=wall_thickness,
        shade_entity_id=w.get("shade_entity_id"),  # Add this line
    )
)
```

### Step 3: Add Data Models

**File:** `sun_hit_detector\core\models.py` (MODIFY)

Add two new dataclasses after the `HitResult` class (around line 160):

```python
@dataclass
class WindowSunDetail:
    """Sun exposure details for a single window.

    Attributes:
        window_id: Window identifier
        is_in_sun: Whether window is receiving direct sunlight
        sun_angle_to_normal_deg: Angle between sun and window normal (0-90 degrees)
        intensity_factor: Relative intensity, cos(angle), range [0, 1]
    """
    window_id: str
    is_in_sun: bool
    sun_angle_to_normal_deg: float
    intensity_factor: float

@dataclass
class WindowSunResult:
    """Result of checking all windows for sun exposure.

    Attributes:
        windows_in_sun: List of window IDs currently receiving sun
        window_details: Detailed info for each window
        sun_azimuth_deg: Sun azimuth used for calculation
        sun_elevation_deg: Sun elevation used for calculation
        sun_direction: Sun direction vector (optional)
        reason: Explanation if no windows in sun (e.g., "sun_below_horizon")
    """
    windows_in_sun: list[str]
    window_details: dict[str, WindowSunDetail]
    sun_azimuth_deg: float
    sun_elevation_deg: float
    sun_direction: Optional[np.ndarray] = None
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "windows_in_sun": self.windows_in_sun,
            "window_details": {
                wid: {
                    "window_id": detail.window_id,
                    "is_in_sun": detail.is_in_sun,
                    "sun_angle_to_normal_deg": round(detail.sun_angle_to_normal_deg, 1),
                    "intensity_factor": round(detail.intensity_factor, 3),
                }
                for wid, detail in self.window_details.items()
            },
            "sun_azimuth_deg": round(self.sun_azimuth_deg, 1),
            "sun_elevation_deg": round(self.sun_elevation_deg, 1),
            "reason": self.reason,
        }
```

### Step 4: Create Window Sun Detection Module

**File:** `sun_hit_detector\core\window_sun.py` (NEW)

Create new module with three main functions:

#### Function 1: `check_window_sun_exposure_geometric(window, sun_direction)`
- Takes: Window object, sun direction vector (from `geometry.sun_direction_simplified()`)
- Returns: `(is_facing_sun: bool, angle_deg: float, intensity: float)`
- Logic:
  1. Compute dot product: `dot(sun_direction, window.normal)`
  2. If dot <= 0: sun behind window, return (False, 90, 0)
  3. Calculate angle: `math.degrees(math.acos(dot_product))`
  4. Intensity = dot_product (cosine of angle)
  5. Return (True, angle, intensity)

#### Function 2: `validate_window_sun_with_ray(window, sun_direction)`
- Takes: Window object, sun direction vector
- Returns: `bool` (True if sun can actually reach window)
- Logic:
  1. Start ray from window center
  2. Cast ray toward sun (using existing `ray_window_intersection()` from `ray_casting.py`)
  3. For thick walls: verify ray passes through window tunnel
  4. Return True if ray path is clear

#### Function 3: `check_windows_from_config(sun_azimuth_deg, sun_elevation_deg, config, use_ray_validation=True)`
- Main API function
- Takes: Sun position, Config object, validation flag
- Returns: `WindowSunResult` object
- Logic:
  1. Check if sun below horizon → return all False with reason
  2. Get wall1_normal from first wall in config
  3. Calculate sun_direction using `sun_direction_simplified()` from `geometry.py`
  4. For each window:
     - Run geometric check
     - If facing sun AND use_ray_validation=True: run ray validation
     - Store results
  5. Build WindowSunResult with all window details
  6. Set windows_in_sun list

### Step 5: Add Home Assistant Service Functions

**File:** `sun_hit_detector\homeassistant\service.py` (MODIFY)

Add four new functions at the end (after line 214):

#### Function 1: `get_window_sun_status(sun_azimuth, sun_elevation, config_path=None)`
- Similar signature to existing `get_sunlight_details()`
- Calls `check_windows_from_config()` with ray validation enabled
- Returns: `result.to_dict()` (JSON-serializable dictionary)
- Uses existing `load_config()` for caching

#### Function 2: `check_window_sunlight(window_id, sun_azimuth, sun_elevation, config_path=None)`
- Simple boolean function for individual window
- Calls `get_window_sun_status()`, checks if `window_id in result['windows_in_sun']`
- Returns: `bool`
- Designed for HA command-line integration

#### Function 3: `get_window_intensity(window_id, sun_azimuth, sun_elevation, config_path=None)`
- Returns intensity factor (0-1) for specific window
- Calls `get_window_sun_status()`, extracts `window_details[window_id]['intensity_factor']`
- Returns: `float` (0.0 if window not in sun)
- Useful for advanced automations (HVAC load calculations)

#### Function 4: `get_shade_sun_info(shade_entity_id, sun_azimuth, sun_elevation, config_path=None)`
- Query by shade entity ID instead of window ID
- Looks up window with matching `shade_entity_id` in config
- Returns: `dict` with window_id, is_in_sun, intensity, angle
- Enables querying: "Does shade X have sun?" without knowing window ID

### Step 6: Extend CLI Script

**File:** `check_plant_sun.py` (MODIFY)

Add new command-line arguments after line 60:

```python
parser.add_argument("--windows", action="store_true",
                   help="Output window sun status instead of plant status")
parser.add_argument("--window-id", type=str,
                   help="Check specific window (with --windows, returns on/off)")
```

Add new logic in `main()` after line 82:

```python
# Handle --windows flag
if args.windows:
    from sun_hit_detector.homeassistant.service import get_window_sun_status, check_window_sunlight

    if args.json:
        # JSON output with all window details
        status = get_window_sun_status(azimuth, elevation, config_path)
        print(json.dumps(status))
    elif args.window_id:
        # Binary on/off for specific window
        is_sunny = check_window_sunlight(args.window_id, azimuth, elevation, config_path)
        print("on" if is_sunny else "off")
    else:
        # Human-readable list of windows in sun
        status = get_window_sun_status(azimuth, elevation, config_path)
        if status['windows_in_sun']:
            print("Windows in sun:", ", ".join(status['windows_in_sun']))
        else:
            reason = status.get('reason', 'No windows receiving sun')
            print(reason)
    sys.exit(0)

# Existing plant logic continues...
```

### Step 7: Create Local Testing Script

**File:** `examples\test_window_sun.py` (NEW)

Create comprehensive test script with:
- Single sun position test (default or --azimuth/--elevation)
- Time-range test (--time-range flag, tests 6am-8pm)
- Table output showing: Time | Azimuth | Elevation | Windows in Sun
- Detailed output: Window ID, In Sun (YES/no), Angle, Intensity

This allows testing locally before deploying to Home Assistant.

**Key features:**
- Auto-calculate sun position from config location
- Override with explicit azimuth/elevation
- Display results in readable table format
- Show intensity factors for debugging

### Step 8: Create Home Assistant Custom Component

**Directory:** `custom_components/sun_shade_integration/` (NEW)

Create a custom HA integration that monitors sun position and updates shade entity attributes.

#### File Structure:
```
custom_components/
  sun_shade_integration/
    __init__.py          # Component initialization & core logic
    manifest.json        # Component metadata
    const.py             # Constants
```

#### File 1: `manifest.json`
```json
{
  "domain": "sun_shade_integration",
  "name": "Sun Shade Integration",
  "version": "1.0.0",
  "documentation": "https://github.com/yourusername/sun-hit-detector",
  "requirements": ["numpy"],
  "codeowners": [],
  "iot_class": "local_polling",
  "dependencies": ["sun"]
}
```

#### File 2: `const.py`
```python
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
```

#### File 3: `__init__.py` (Core Logic)
```python
"""Sun Shade Integration for Home Assistant."""
import logging
from datetime import timedelta
from pathlib import Path

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE

from .const import DOMAIN, CONF_CONFIG_PATH, CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL
from .const import ATTR_WINDOW_ID, ATTR_WINDOW_HAS_SUN, ATTR_SUN_INTENSITY, ATTR_SUN_ANGLE

_LOGGER = logging.getLogger(__name__)

async def async_setup(hass: HomeAssistant, config: dict):
    """Set up the sun shade integration."""
    domain_config = config.get(DOMAIN, {})
    config_path = domain_config.get(CONF_CONFIG_PATH, "/config/sun_plant_config.json")
    update_interval = domain_config.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)

    # Import sun simulator modules
    import sys
    sys.path.insert(0, "/sun-hit-detector")
    from sun_hit_detector.core.models import Config
    from sun_hit_detector.core.window_sun import check_windows_from_config

    # Load config
    try:
        sim_config = Config.from_json_file(config_path)
        _LOGGER.info(f"Loaded sun simulator config from {config_path}")
    except Exception as e:
        _LOGGER.error(f"Failed to load config: {e}")
        return False

    # Build window → shade mapping
    window_to_shade = {}
    for window in sim_config.windows:
        if window.shade_entity_id:
            window_to_shade[window.id] = window.shade_entity_id

    _LOGGER.info(f"Found {len(window_to_shade)} windows with shade mappings")

    @callback
    async def update_shade_sun_attributes(now=None):
        """Update sun attributes on shade entities."""
        # Get sun position from sun.sun entity
        sun_entity = hass.states.get("sun.sun")
        if not sun_entity or sun_entity.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            _LOGGER.warning("Sun entity not available")
            return

        azimuth = sun_entity.attributes.get("azimuth")
        elevation = sun_entity.attributes.get("elevation")

        if azimuth is None or elevation is None:
            _LOGGER.warning("Sun position attributes missing")
            return

        # Run window sun calculation
        try:
            result = await hass.async_add_executor_job(
                check_windows_from_config,
                azimuth,
                elevation,
                sim_config
            )

            # Update shade entity attributes
            for window_id, shade_entity_id in window_to_shade.items():
                window_detail = result.window_details.get(window_id)
                if not window_detail:
                    continue

                # Get current shade entity state
                shade_entity = hass.states.get(shade_entity_id)
                if not shade_entity:
                    _LOGGER.warning(f"Shade entity {shade_entity_id} not found")
                    continue

                # Merge existing attributes with new window sun attributes
                new_attributes = dict(shade_entity.attributes)
                new_attributes[ATTR_WINDOW_ID] = window_id
                new_attributes[ATTR_WINDOW_HAS_SUN] = window_detail.is_in_sun
                new_attributes[ATTR_SUN_INTENSITY] = round(window_detail.intensity_factor, 3)
                new_attributes[ATTR_SUN_ANGLE] = round(window_detail.sun_angle_to_normal_deg, 1)

                # Update entity state with new attributes
                hass.states.async_set(
                    shade_entity_id,
                    shade_entity.state,
                    new_attributes
                )

            _LOGGER.debug(f"Updated {len(window_to_shade)} shade entities with sun data")

        except Exception as e:
            _LOGGER.error(f"Error updating shade sun attributes: {e}")

    # Schedule periodic updates
    async_track_time_interval(
        hass,
        update_shade_sun_attributes,
        timedelta(seconds=update_interval)
    )

    # Run initial update
    await update_shade_sun_attributes()

    return True
```

#### Configuration in `configuration.yaml`:
```yaml
sun_shade_integration:
  config_path: /config/sun_plant_config.json
  update_interval: 300  # seconds (5 minutes)
```

#### Automation Example:
Now automations use the existing shade entities directly:

```yaml
automation:
  - alias: "Auto-close shade when window gets direct sun"
    trigger:
      - platform: state
        entity_id: cover.living_room_shade_1a
        attribute: window_has_sun
        to: true
    condition:
      - condition: template
        value_template: "{{ state_attr('cover.living_room_shade_1a', 'sun_intensity') | float > 0.5 }}"
    action:
      - service: cover.close_cover
        target:
          entity_id: cover.living_room_shade_1a
```

### Step 9: Create Documentation

**File:** `WINDOW_SUN_TRACKING.md` (NEW)

Document:
- Feature overview and use cases
- How it works (geometric + ray-based approach)
- CLI usage examples
- Python API usage examples
- Home Assistant integration examples
- Configuration for all 8 windows and shade mappings
- Template sensor setup
- Automation examples (close shades when window gets sun, adjust HVAC)
- How to query shade sun status in templates
- Troubleshooting common issues

## Critical Files

1. **`config\default_config.json`** - Add shade_entity_id mappings to window definitions
2. **`sun_hit_detector\core\models.py`** - Add shade_entity_id field to Window class, add WindowSunResult and WindowSunDetail classes
3. **`sun_hit_detector\core\window_sun.py`** - New module with all window sun detection logic
4. **`sun_hit_detector\homeassistant\service.py`** - Add HA service functions including get_shade_sun_info()
5. **`check_plant_sun.py`** - Extend CLI with --windows flag
6. **`examples\test_window_sun.py`** - Local testing script
7. **`custom_components\sun_shade_integration\__init__.py`** - Custom HA component that updates shade entity attributes
8. **`custom_components\sun_shade_integration\manifest.json`** - Component metadata
9. **`custom_components\sun_shade_integration\const.py`** - Component constants

## Sun Service Integration Details

### How the Component Monitors Sun Position

**Home Assistant's sun.sun Entity:**
- HA automatically creates a `sun.sun` entity with real-time sun position
- Attributes updated continuously based on:
  - System time
  - GPS coordinates (from HA configuration)
  - Astronomical calculations (NOAA algorithm)
- Key attributes:
  - `azimuth`: 0-360° (0=North, 90=East, 180=South, 270=West)
  - `elevation`: -90 to +90° (negative = below horizon)
  - `rising`: boolean (is sun currently rising)
  - `next_dawn`, `next_dusk`, `next_sunrise`, `next_sunset`: timestamps

**Integration Approach:**

The custom component uses **time-based polling** (every 5 minutes):

```python
async_track_time_interval(
    hass,
    update_shade_sun_attributes,  # Callback function
    timedelta(seconds=300)         # 5 minute interval
)
```

**Why not state change listener?**
- `sun.sun` attributes update continuously (every second)
- Would cause excessive calculations
- Sun position changes slowly - 5 minute polling is sufficient

**Alternative: State Change Listener (if needed)**
```python
@callback
def sun_changed(event):
    """Handle sun position changes."""
    # Only update if azimuth or elevation changed significantly
    old_azimuth = event.data.get("old_state").attributes.get("azimuth")
    new_azimuth = event.data.get("new_state").attributes.get("azimuth")

    if abs(new_azimuth - old_azimuth) > 1:  # 1 degree threshold
        await update_shade_sun_attributes()

async_track_state_change_event(
    hass,
    "sun.sun",
    sun_changed
)
```

**Data Flow:**
1. Timer triggers every 5 minutes
2. Component reads `hass.states.get("sun.sun")`
3. Extracts `azimuth` and `elevation` attributes
4. Calls our Python ray calculator: `check_windows_from_config(azimuth, elevation, config)`
5. Returns `WindowSunResult` with all window data
6. Component loops through shade entities
7. Uses `hass.states.async_set()` to update each shade's attributes
8. HA UI automatically refreshes to show new attribute values

## Dependencies

- Uses existing modules:
  - `geometry.py`: `sun_direction_simplified()`, `dot()`, `angle_between_vectors()`
  - `ray_casting.py`: `ray_window_intersection()` for ray validation
  - `models.py`: Window, Config classes
- HA dependencies:
  - `sun` integration (built-in, provides `sun.sun` entity)
  - No external Python packages beyond numpy (already in requirements)
- No new external dependencies required

## Testing Strategy

### Local Testing (Before Home Assistant)

1. **Unit Tests**:
   - Test geometric check with known angles (0°, 45°, 90°)
   - Test ray validation with different wall thicknesses
   - Test sun below horizon case

2. **Integration Tests**:
   ```bash
   # Test with explicit sun position
   python check_plant_sun.py 210 30 --windows --json

   # Test specific window
   python check_plant_sun.py 210 30 --windows --window-id window_1a

   # Test with example script
   python examples/test_window_sun.py
   python examples/test_window_sun.py --time-range
   ```

3. **Visual Verification**:
   - Run time-range test over full day
   - Verify windows transition at expected times based on wall orientations
   - Check that wall_1 windows (facing 210°) get afternoon sun
   - Check that wall_2 windows (facing 307°) get different timing

### Home Assistant Testing (After Deployment)

1. **CLI in Docker**:
   ```bash
   ssh dell7050
   docker exec home-assistant python3 /sun-hit-detector/check_plant_sun.py \
     $(date) --config /config/sun_plant_config.json --windows --json
   ```

2. **Sensor Verification**:
   - Developer Tools → States
   - Check `binary_sensor.window_1a_sunlight` etc.
   - Verify updates every 5 minutes
   - Check attributes on `sensor.window_sun_status`

3. **Automation Testing**:
   - Create test automation
   - Trigger manually
   - Verify actions execute correctly

## Deployment

### Local Development (Steps 1-5)

Implement and test all code changes locally on Windows machine.

### Server Deployment

1. **Copy Core Modules**:
```bash
# Core modules
scp sun_hit_detector/core/models.py dell7050:/home/master/sun-hit-detector/sun_hit_detector/core/
scp sun_hit_detector/core/window_sun.py dell7050:/home/master/sun-hit-detector/sun_hit_detector/core/

# Home Assistant integration (for CLI support)
scp sun_hit_detector/homeassistant/service.py dell7050:/home/master/sun-hit-detector/sun_hit_detector/homeassistant/

# CLI (for standalone testing)
scp check_plant_sun.py dell7050:/home/master/sun-hit-detector/
```

2. **Deploy Custom Component**:
```bash
# Create custom_components directory if it doesn't exist
ssh dell7050 "sudo mkdir -p /home/master/homeassistant/custom_components/sun_shade_integration"

# Copy custom component files
scp -r custom_components/sun_shade_integration/* dell7050:/tmp/sun_shade_integration/
ssh dell7050 "sudo mv /tmp/sun_shade_integration/* /home/master/homeassistant/custom_components/sun_shade_integration/"
ssh dell7050 "sudo chown -R 1000:1000 /home/master/homeassistant/custom_components"
```

3. **Update Config with Shade Entity IDs**:
```bash
# Update sun_plant_config.json with shade entity IDs
scp config/default_config.json dell7050:/tmp/sun_plant_config.json
ssh -t dell7050 "sudo mv /tmp/sun_plant_config.json /home/master/homeassistant/sun_plant_config.json"
```

4. **Update HA Configuration**:
```bash
# Edit configuration.yaml to enable the integration
ssh -t dell7050 "sudo nano /home/master/homeassistant/configuration.yaml"
```

Add to `configuration.yaml`:
```yaml
sun_shade_integration:
  config_path: /config/sun_plant_config.json
  update_interval: 300
```

5. **Restart Home Assistant**:
```bash
ssh dell7050 "cd /home/master && docker compose restart home-assistant"
```

6. **Verify**:
```bash
# Check logs for component loading
ssh dell7050 "docker logs -f home-assistant | grep sun_shade"

# Check shade entity attributes in HA Developer Tools → States
# Look for: window_id, window_has_sun, sun_intensity, sun_angle_deg
```

## Verification Plan

### Phase 1: Local Testing (Before HA Deployment)

1. ✅ **Core Functionality**:
   ```bash
   # Test CLI with explicit sun position
   python check_plant_sun.py 210 30 --windows --json

   # Test with time range
   python examples/test_window_sun.py --time-range
   ```
   - Verify correct window IDs in sun at different times
   - Check intensity factors (perpendicular = 1.0, oblique < 1.0)
   - Confirm ray validation detects obstructions

2. ✅ **Config with Shade Mappings**:
   ```bash
   # Add shade_entity_id to each window in config
   # Verify config loads correctly
   python -c "from sun_hit_detector.core.models import Config; c = Config.from_json_file('config/default_config.json'); print([w.shade_entity_id for w in c.windows])"
   ```

### Phase 2: Home Assistant Testing (After Deployment)

1. ✅ **Component Loading**:
   - Check HA logs: `docker logs home-assistant | grep sun_shade`
   - Verify component loaded without errors
   - Confirm config file read successfully

2. ✅ **Shade Entity Attributes**:
   - Go to Developer Tools → States
   - Find shade entity (e.g., `cover.living_room_shade_1a`)
   - Verify new attributes present:
     - `window_id`: "window_1a"
     - `window_has_sun`: true/false
     - `sun_intensity`: 0.0-1.0
     - `sun_angle_deg`: 0-90

3. ✅ **Real-Time Updates**:
   - Watch shade entity attributes over time
   - Verify updates every 5 minutes
   - Confirm sun position changes reflect in attributes
   - Check windows transition from sun→no sun as expected

4. ✅ **Automation Testing**:
   ```yaml
   # Test automation
   automation:
     - alias: "Test shade sun detection"
       trigger:
         - platform: state
           entity_id: cover.living_room_shade_1a
           attribute: window_has_sun
           to: true
       action:
         - service: notify.persistent_notification
           data:
             message: "Shade 1A window has sun!"
   ```
   - Manually test or wait for sun to hit window
   - Verify automation triggers correctly
   - Check notification appears

5. ✅ **Edge Cases**:
   - Sun below horizon: all windows should have `window_has_sun: false`
   - Dawn/dusk transitions: verify smooth updates
   - Multiple windows with sun simultaneously: check all update correctly

## Implementation Notes

- The geometric check acts as a fast pre-filter (O(n) where n = number of windows)
- Ray validation runs only for windows facing sun (typically 2-4 windows)
- Total computation time: <10ms for 8 windows on typical hardware
- Config caching (already in `service.py`) prevents reloading JSON on each call
- Exit codes always 0 for Home Assistant compatibility (errors print "off")

## Future Enhancements

- Add optional `--no-ray-validation` flag for faster geometric-only mode
- Add window obstruction detection (one window shadowing another)
- Track historical sun patterns per window
- Integration with HVAC automation (solar heat gain calculations)
- Visualization: 3D scene showing which windows have sun

## Trade-offs

**Chosen: Ray-Based Validation**
- Pro: Accurate, accounts for obstructions, handles thick walls correctly
- Con: Slightly slower than pure geometric (but still <10ms)
- Justification: User prioritized accuracy, and performance impact is negligible

**Chosen: Custom Component Directly Extending Shade Entities**
- Pro: Most elegant - attributes added directly to existing entities, no wrappers needed
- Pro: Automatic updates via sun.sun state tracking
- Pro: Native HA integration, appears as if shades always had sun attributes
- Con: Requires custom component deployment (more complex than YAML config)
- Justification: User prioritized elegance and clean UX, willing to use custom component

**Chosen: Track Intensity**
- Pro: Enables advanced use cases (HVAC load, solar heat gain)
- Con: Adds complexity to data model
- Justification: User's preference, minimal code impact
