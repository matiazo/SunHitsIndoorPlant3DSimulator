# Home Assistant Deployment Guide

## Current Status

✅ **Configuration Updated** - All 8 windows are now mapped to Overkiz shade entities

### Window to Shade Mapping

#### Wall 1 (Front) - Azimuth 210° - INVERTED ORDER
| Window ID | Position | Shade Entity ID | Notes |
|-----------|----------|-----------------|-------|
| window_1a | x=0.36 | `cover.living_room_front_shade_4` | Shade 4 closest to corner |
| window_1b | x=1.44 | `cover.living_room_front_shade_3` | |
| window_1c | x=2.52 | `cover.living_room_front_shade_2` | |
| window_1d | x=3.60 | `cover.living_room_front_shade_1` | Shade 1 farthest from corner |

#### Wall 2 (Side) - Azimuth 300° - NORMAL ORDER
| Window ID | Position | Shade Entity ID | Notes |
|-----------|----------|-----------------|-------|
| window_2a | y=0.80 | `cover.living_room_side_shade_1` | Closest to corner |
| window_2b | y=4.00 | `cover.living_room_side_shade_2` | |
| window_2c | y=5.07 | `cover.living_room_side_shade_3` | |
| window_2d | y=8.26 | `cover.living_room_side_shade_4` | Farthest from corner |

---

## Deployment Steps

### Phase 1: Local Testing (Windows Machine)

Test the implementation locally before deploying to Home Assistant:

```bash
# Test window sun detection with current sun position
python check_plant_sun.py --windows --json

# Test specific window
python check_plant_sun.py 210 30 --windows --window-id window_1a

# Run comprehensive testing script
python examples/test_window_sun.py --time-range
```

### Phase 2: Copy Files to Server

Copy the updated files to your dell7050 server:

```bash
# Core simulator modules
scp sun_hit_detector/core/models.py dell7050:/home/master/sun-hit-detector/sun_hit_detector/core/
scp sun_hit_detector/core/window_sun.py dell7050:/home/master/sun-hit-detector/sun_hit_detector/core/

# Home Assistant service functions
scp sun_hit_detector/homeassistant/service.py dell7050:/home/master/sun-hit-detector/sun_hit_detector/homeassistant/

# CLI script
scp check_plant_sun.py dell7050:/home/master/sun-hit-detector/

# Updated config with correct shade entity IDs
scp config/default_config.json dell7050:/tmp/sun_plant_config.json
ssh -t dell7050 "sudo mv /tmp/sun_plant_config.json /home/master/homeassistant/sun_plant_config.json"
```

### Phase 3: Deploy Custom Component

Copy the custom Home Assistant component:

```bash
# Create directory structure
ssh dell7050 "sudo mkdir -p /home/master/homeassistant/custom_components/sun_shade_integration"

# Copy component files
scp custom_components/sun_shade_integration/manifest.json dell7050:/tmp/
scp custom_components/sun_shade_integration/const.py dell7050:/tmp/
scp custom_components/sun_shade_integration/__init__.py dell7050:/tmp/

# Move to correct location with proper permissions
ssh dell7050 "sudo mv /tmp/manifest.json /home/master/homeassistant/custom_components/sun_shade_integration/"
ssh dell7050 "sudo mv /tmp/const.py /home/master/homeassistant/custom_components/sun_shade_integration/"
ssh dell7050 "sudo mv /tmp/__init__.py /home/master/homeassistant/custom_components/sun_shade_integration/"
ssh dell7050 "sudo chown -R 1000:1000 /home/master/homeassistant/custom_components"
```

### Phase 4: Configure Home Assistant

1. **Edit configuration.yaml:**

```bash
ssh -t dell7050 "sudo nano /home/master/homeassistant/configuration.yaml"
```

2. **Add the integration:**

```yaml
sun_shade_integration:
  config_path: /config/sun_plant_config.json
  update_interval: 300  # 5 minutes
```

3. **Save and exit** (Ctrl+X, Y, Enter)

### Phase 5: Restart Home Assistant

```bash
# Restart the container
ssh dell7050 "cd /home/master && docker compose restart home-assistant"

# Monitor logs for component loading
ssh dell7050 "docker logs -f home-assistant | grep -i 'sun_shade'"
```

Expected log output:
```
INFO (MainThread) [custom_components.sun_shade_integration] Loaded sun simulator config from /config/sun_plant_config.json
INFO (MainThread) [custom_components.sun_shade_integration] Found 8 windows with shade mappings
INFO (MainThread) [custom_components.sun_shade_integration] Sun shade integration initialized. Updates every 300 seconds.
```

### Phase 6: Verify Integration

1. **Check Developer Tools → States** for shade entities:
   - `cover.living_room_front_shade_1` through `cover.living_room_front_shade_4`
   - `cover.living_room_side_shade_1` through `cover.living_room_side_shade_4`

2. **Verify new attributes appear on each shade:**
   ```yaml
   window_id: "window_1a"
   window_has_sun: true/false
   sun_intensity: 0.0-1.0
   sun_angle_deg: 0-90
   ```

3. **Test CLI from inside container:**
   ```bash
   ssh dell7050
   docker exec home-assistant python3 /sun-hit-detector/check_plant_sun.py \
     $(date +"%Y-%m-%d %H:%M:%S") \
     --config /config/sun_plant_config.json \
     --windows --json
   ```

---

## Example Automations

### Auto-Close Shades When Window Gets Sun

```yaml
automation:
  - alias: "Close front shade 4 when window gets direct sun"
    trigger:
      - platform: state
        entity_id: cover.living_room_front_shade_4
        attribute: window_has_sun
        to: true
    condition:
      - condition: template
        value_template: "{{ state_attr('cover.living_room_front_shade_4', 'sun_intensity') | float > 0.5 }}"
      - condition: time
        after: "10:00:00"
        before: "18:00:00"
    action:
      - service: cover.close_cover
        target:
          entity_id: cover.living_room_front_shade_4
```

### Smart Shade Positioning Based on Intensity

```yaml
automation:
  - alias: "Adjust shade position based on sun intensity"
    trigger:
      - platform: state
        entity_id: cover.living_room_side_shade_3
        attribute: sun_intensity
    condition:
      - condition: state
        entity_id: cover.living_room_side_shade_3
        attribute: window_has_sun
        state: true
    action:
      - service: cover.set_cover_position
        target:
          entity_id: cover.living_room_side_shade_3
        data:
          position: >
            {% set intensity = state_attr('cover.living_room_side_shade_3', 'sun_intensity') | float %}
            {{ 100 - (intensity * 100) | int }}
```

### All-Windows Dashboard Card

```yaml
type: entities
title: Window Sun Exposure
entities:
  - entity: cover.living_room_front_shade_1
    secondary_info: >
      {% if state_attr('cover.living_room_front_shade_1', 'window_has_sun') %}
        ☀️ Sun: {{ state_attr('cover.living_room_front_shade_1', 'sun_intensity') | round(2) }}
      {% else %}
        🌙 No sun
      {% endif %}
  - entity: cover.living_room_front_shade_2
    secondary_info: >
      {% if state_attr('cover.living_room_front_shade_2', 'window_has_sun') %}
        ☀️ Sun: {{ state_attr('cover.living_room_front_shade_2', 'sun_intensity') | round(2) }}
      {% else %}
        🌙 No sun
      {% endif %}
  # ... repeat for all 8 shades
```

---

## Troubleshooting

### Component Not Loading

**Check logs:**
```bash
ssh dell7050 "docker logs home-assistant | grep -A 10 sun_shade"
```

**Common issues:**
- Config file path incorrect: Check `/config/sun_plant_config.json` exists
- Python path issue: Verify `/sun-hit-detector` is mounted in container
- Syntax error in configuration.yaml: Validate YAML formatting

### Attributes Not Appearing

**Verify shade entity IDs exist:**
```bash
ssh dell7050 "docker exec home-assistant ha-cli entity list | grep living.*shade"
```

**Check if names match exactly:**
- Overkiz shades may have slightly different naming
- Use Developer Tools → States to find exact entity IDs
- Update config/default_config.json if names differ

### Updates Not Happening

**Check sun.sun entity:**
```bash
ssh dell7050 "docker exec home-assistant ha-cli state get sun.sun"
```

Should show `azimuth` and `elevation` attributes.

**Force manual update:**
```bash
# Restart integration
ssh dell7050 "cd /home/master && docker compose restart home-assistant"
```

---

## Testing Commands

### Test current sun position:
```bash
python check_plant_sun.py --windows --json
```

### Test specific window:
```bash
python check_plant_sun.py 210 30 --windows --window-id window_2c
```

### Run year simulation:
```bash
python examples/simulate_yearly_plant_sun.py
```

### Get shade info by entity ID:
```python
from sun_hit_detector.homeassistant.service import get_shade_sun_info

info = get_shade_sun_info(
    "cover.living_room_front_shade_4",
    sun_azimuth=210,
    sun_elevation=30
)
print(info)
```

---

## File Locations

### On Windows Machine (Development):
- Config: `c:\repo\SunHitsIndoorPlant3DSimulator\config\default_config.json`
- Custom component: `c:\repo\SunHitsIndoorPlant3DSimulator\custom_components\sun_shade_integration\`

### On Server (Production):
- Simulator: `/home/master/sun-hit-detector/`
- Config: `/home/master/homeassistant/sun_plant_config.json`
- Custom component: `/home/master/homeassistant/custom_components/sun_shade_integration/`

---

## Next Steps

1. ✅ Config updated with correct Overkiz shade entity IDs
2. ⏳ Test locally on Windows machine
3. ⏳ Deploy to dell7050 server
4. ⏳ Configure Home Assistant
5. ⏳ Restart and verify
6. ⏳ Create automations

**Ready to deploy!** Follow the steps above to deploy the window sun integration to your Home Assistant instance.
