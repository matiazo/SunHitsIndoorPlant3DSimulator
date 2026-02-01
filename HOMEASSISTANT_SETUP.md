# Home Assistant Integration Setup

## Server Access

- **Server**: `dell7050` (accessible via SSH)
- **SSH Command**: `ssh dell7050`
- **Home Assistant Config Path**: `/home/master/homeassistant/`
- **Docker Compose Location**: `/home/master/docker-compose.yml`

## Home Assistant Details

- **Container Name**: `home-assistant`
- **Image**: `ghcr.io/home-assistant/home-assistant:2025.9.3`
- **Network Mode**: `host`
- **Config Mount**: `/home/master/homeassistant:/config`
- **Timezone**: `America/New_York`

## Sun Plant Simulator Integration

### Installation Location
- **Host Path**: `/home/master/sun-plant-simulator/`
- **Container Mount**: `/sun-plant-simulator:ro`
- **Virtual Environment**: `/home/master/sun-plant-simulator/venv/`

### Configuration Files
- **Plant/Room Config**: `/home/master/homeassistant/sun_plant_config.json` (copy of `config/default_config.json`)
- **Sensor Config**: `/home/master/homeassistant/sun_plant_sensor.yaml`
- **Main Config**: `/home/master/homeassistant/configuration.yaml` (includes `command_line: !include sun_plant_sensor.yaml`)

### Sensors Created
1. **Binary Sensor**: `binary_sensor.plant_direct_sunlight`
   - State: `on`/`off`
   - Device Class: `light`
   - Scan Interval: 300 seconds

2. **Sensor**: `sensor.plant_sun_details`
   - Value: `true`/`false` (is_hit)
   - Attributes: `sun_azimuth`, `sun_elevation`, `hit_window`, `hit_wall`, `timestamp`

## Common Commands

### Testing the Script Inside Container
```bash
ssh dell7050 "docker exec home-assistant python3 /sun-plant-simulator/check_plant_sun.py --config /config/sun_plant_config.json"
```

### Testing with JSON Output
```bash
ssh dell7050 "docker exec home-assistant python3 /sun-plant-simulator/check_plant_sun.py --config /config/sun_plant_config.json --json"
```

### View Home Assistant Logs
```bash
ssh dell7050 "docker logs home-assistant --tail 50"
```

### Restart Home Assistant
```bash
ssh dell7050 "cd /home/master && docker compose restart home-assistant"
```

### Recreate Home Assistant Container
```bash
ssh dell7050 "cd /home/master && docker compose down home-assistant && docker compose up -d home-assistant"
```

### Copy Files to Server
```bash
scp <local_file> dell7050:/home/master/sun-plant-simulator/
```

### Copy Config to Home Assistant (requires sudo)
```bash
ssh -t dell7050  # Interactive session for sudo
sudo cp /home/master/sun-plant-simulator/config/default_config.json /home/master/homeassistant/sun_plant_config.json
```

## Updating the Integration

1. **Update Python files locally** in VS Code
2. **Copy changed files to server**:
   ```bash
   scp check_plant_sun.py dell7050:/home/master/sun-plant-simulator/
   scp sun_plant_simulator/core/models.py dell7050:/home/master/sun-plant-simulator/sun_plant_simulator/core/
   ```
3. **Test the script**:
   ```bash
   ssh dell7050 "docker exec home-assistant python3 /sun-plant-simulator/check_plant_sun.py --config /config/sun_plant_config.json"
   ```
4. **Restart Home Assistant** if configuration changed:
   ```bash
   ssh dell7050 "cd /home/master && docker compose restart home-assistant"
   ```

## Directory Structure on Server

```
/home/master/
├── docker-compose.yml          # Docker services config
├── homeassistant/              # HA config directory (mounted as /config)
│   ├── configuration.yaml      # Main HA config
│   ├── sun_plant_sensor.yaml   # Command-line sensor definitions
│   ├── sun_plant_config.json   # Plant/room geometry config
│   ├── automations.yaml
│   └── ...
└── sun-plant-simulator/        # This project (mounted as /sun-plant-simulator)
    ├── check_plant_sun.py      # CLI entry point for HA
    ├── config/
    │   └── default_config.json
    ├── sun_plant_simulator/
    │   ├── core/
    │   └── homeassistant/
    └── venv/                   # Python virtual environment
```

## Troubleshooting

### Permission Denied on Config Files
Home Assistant config directory is owned by root. Use `sudo` via interactive SSH:
```bash
ssh -t dell7050
sudo cp <source> /home/master/homeassistant/<dest>
```

### Script Returns Exit Code 1
The script should always return exit code 0 for Home Assistant compatibility. If it returns 1, check:
- Import errors (run script manually to see traceback)
- Missing config file
- Syntax errors in Python files

### Sensor Not Updating
- Check scan_interval (default 300 seconds = 5 minutes)
- Verify command works: `docker exec home-assistant python3 /sun-plant-simulator/check_plant_sun.py --config /config/sun_plant_config.json`
- Check HA logs: `docker logs home-assistant | grep command_line`

### Container Crash Loop
If HA keeps crashing, check logs for Python package issues:
```bash
ssh dell7050 "docker logs home-assistant 2>&1 | tail -50"
```
Try recreating the container: `docker compose down home-assistant && docker compose up -d home-assistant`
