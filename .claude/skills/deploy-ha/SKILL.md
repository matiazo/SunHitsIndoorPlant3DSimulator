---
name: deploy-ha
description: Deploy sun shade integration to Home Assistant on dell7050 via SCP and restart the container
disable-model-invocation: true
---

# Deploy Sun Shade Integration to Home Assistant

Deploy the custom component and simulator core to the dell7050 Home Assistant server.

## Arguments

- `--component-only` — Only deploy the custom_components directory (skip core simulator)
- `--core-only` — Only deploy the sun_hit_detector core modules (skip custom component)
- `--config` — Also deploy config/default_config.json
- `--no-restart` — Skip the HA container restart after deploying
- `--dry-run` — Show what would be deployed without actually doing it

## Deployment Steps

### 1. Pre-flight checks

Run the test suite first to ensure nothing is broken:

```bash
cd C:/repo/SunHitsIndoorPlant3DSimulator && python -m pytest tests/ -x -q
```

If tests fail, stop and report the failures. Do NOT deploy broken code.

### 2. Deploy core simulator modules (unless --component-only)

```bash
scp sun_hit_detector/core/models.py dell7050:/home/master/sun-hit-detector/sun_hit_detector/core/
scp sun_hit_detector/core/window_sun.py dell7050:/home/master/sun-hit-detector/sun_hit_detector/core/
scp sun_hit_detector/core/coordinates.py dell7050:/home/master/sun-hit-detector/sun_hit_detector/core/
scp sun_hit_detector/core/geometry.py dell7050:/home/master/sun-hit-detector/sun_hit_detector/core/
scp sun_hit_detector/core/hit_test.py dell7050:/home/master/sun-hit-detector/sun_hit_detector/core/
scp sun_hit_detector/core/ray_casting.py dell7050:/home/master/sun-hit-detector/sun_hit_detector/core/
scp sun_hit_detector/core/sun_position.py dell7050:/home/master/sun-hit-detector/sun_hit_detector/core/
scp sun_hit_detector/core/__init__.py dell7050:/home/master/sun-hit-detector/sun_hit_detector/core/
```

### 3. Deploy custom component (unless --core-only)

```bash
# Ensure target directory exists
ssh dell7050 "sudo mkdir -p /home/master/homeassistant/custom_components/sun_shade_integration"

# Copy all component files via /tmp to handle permissions
for f in __init__.py binary_sensor.py config_flow.py const.py manifest.json sensor.py strings.json; do
  scp custom_components/sun_shade_integration/$f dell7050:/tmp/
  ssh dell7050 "sudo mv /tmp/$f /home/master/homeassistant/custom_components/sun_shade_integration/"
done

# Fix ownership
ssh dell7050 "sudo chown -R 1000:1000 /home/master/homeassistant/custom_components"
```

### 4. Deploy config (only if --config flag)

```bash
scp config/default_config.json dell7050:/tmp/sun_plant_config.json
ssh dell7050 "sudo mv /tmp/sun_plant_config.json /home/master/homeassistant/sun_plant_config.json"
```

### 5. Restart Home Assistant (unless --no-restart)

```bash
ssh dell7050 "cd /home/master && docker compose restart home-assistant"
```

### 6. Verify

Wait a few seconds after restart, then check logs:

```bash
ssh dell7050 "docker logs --tail 30 home-assistant 2>&1 | grep -i sun_shade"
```

Report the deployment result: which files were copied, whether restart succeeded, and any log output from the integration.

## SSH Details

- **Host**: dell7050 (192.168.7.210)
- **SSH key**: `~/.ssh/kube1_id_rsa`
- **User**: `master`
- **HA container name**: `home-assistant`
- **Simulator path on server**: `/home/master/sun-hit-detector/`
- **Component path on server**: `/home/master/homeassistant/custom_components/sun_shade_integration/`
