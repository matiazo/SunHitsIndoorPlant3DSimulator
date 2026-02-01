"""Interactive visualization with time slider.

This module creates an HTML visualization with a time slider to explore
how sunlight hits the plant throughout the day.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from ..core.geometry import sun_direction_from_angles
from ..core.hit_test import check_sun_hits_plant, generate_plant_sample_points
from ..core.models import Config, Plant, Window


def create_time_slider_visualization(
    config: Config,
    sun_data: Optional[list[dict]] = None,
    output_path: Optional[str] = None,
    date_str: Optional[str] = None,
) -> str:
    """Create an interactive HTML visualization with time slider.

    Args:
        config: Room configuration.
        sun_data: List of {timestamp, azimuth_deg, elevation_deg} dicts.
                  If None, generates hourly data for a sample day.
        output_path: Path to save HTML file. If None, returns HTML string.
        date_str: Date string to display (e.g., "2024-06-21").

    Returns:
        HTML string or path to saved file.
    """
    # Generate default sun data if not provided
    if sun_data is None:
        sun_data = generate_sample_sun_data()

    if date_str is None:
        from datetime import date
        date_str = date.today().strftime("%Y-%m-%d")

    # Pre-compute hit results for all timestamps
    results = []
    for point in sun_data:
        hit = check_sun_hits_plant(
            sun_azimuth_deg=point["azimuth_deg"],
            sun_elevation_deg=point["elevation_deg"],
            plant=config.plant,
            windows=config.windows,
        )
        results.append({
            "timestamp": point["timestamp"],
            "azimuth": point["azimuth_deg"],
            "elevation": point["elevation_deg"],
            "is_hit": hit.is_hit,
            "window_id": hit.window_id,
            "hit_points": [p.tolist() for p in hit.hit_points] if hit.hit_points else [],
            "sun_direction": hit.sun_direction.tolist() if hit.sun_direction is not None else None,
        })

    # Build the HTML with embedded data and Plotly
    html = build_interactive_html(config, results, date_str)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path
    return html


def generate_sample_sun_data() -> list[dict]:
    """Generate sample sun position data for a day."""
    data = []
    # Simulate sun path for a mid-latitude summer day
    for hour in range(5, 21):  # 5 AM to 8 PM
        for minute in [0, 30]:
            if hour == 20 and minute == 30:
                continue

            time_decimal = hour + minute / 60

            # Simple sun path model
            # Solar noon around 12:00
            hour_angle = (time_decimal - 12) * 15  # degrees from noon

            # Approximate azimuth (east in morning, west in afternoon)
            if time_decimal < 12:
                azimuth = 90 + (12 - time_decimal) * 7.5  # morning: east to south
            else:
                azimuth = 180 + (time_decimal - 12) * 7.5  # afternoon: south to west

            # Approximate elevation (peaks at noon)
            max_elevation = 65  # summer max
            elevation = max_elevation * math.cos(math.radians(hour_angle))

            if elevation > 0:
                data.append({
                    "timestamp": f"{hour:02d}:{minute:02d}",
                    "azimuth_deg": azimuth,
                    "elevation_deg": elevation,
                })

    return data


def build_interactive_html(config: Config, results: list[dict], date_str: str = "") -> str:
    """Build the complete interactive HTML page."""

    # Convert config to JSON for JavaScript
    config_json = json.dumps({
        "plant": {
            "center_x": config.plant.center_x,
            "center_y": config.plant.center_y,
            "radius": config.plant.radius,
            "z_min": config.plant.z_min,
            "z_max": config.plant.z_max,
        },
        "walls": [
            {
                "id": wall.id,
                "normal_azimuth": wall.outward_normal_azimuth_deg,
                "thickness": next((w.wall_thickness for w in config.windows if w.wall_id == wall.id), 0.3),
            }
            for wall in config.walls
        ],
        "windows": [
            {
                "id": w.id,
                "wall_id": w.wall_id,
                "center": w.center.tolist(),
                "width": w.width,
                "height": w.height,
                "wall_normal_azimuth": w.wall_normal_azimuth,
                "wall_thickness": w.wall_thickness,
            }
            for w in config.windows
        ],
    })

    results_json = json.dumps(results)

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Sun-Plant Simulation</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .controls {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .time-display {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .status {{
            font-size: 24px;
            padding: 10px 20px;
            border-radius: 4px;
            display: inline-block;
            margin-bottom: 15px;
        }}
        .status.hit {{
            background: #4CAF50;
            color: white;
        }}
        .status.miss {{
            background: #9E9E9E;
            color: white;
        }}
        .info {{
            color: #666;
            font-size: 14px;
        }}
        .slider-container {{
            margin: 20px 0;
        }}
        #timeSlider {{
            width: 100%;
            height: 30px;
            cursor: pointer;
        }}
        .plot-container {{
            background: white;
            border-radius: 8px;
            padding: 10px;
        }}
        /* Config panel styles */
        .config-panel {{
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .config-header {{
            background: #f0f0f0;
            padding: 12px 20px;
            cursor: pointer;
            border-radius: 8px 8px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: bold;
            color: #333;
        }}
        .config-header:hover {{
            background: #e8e8e8;
        }}
        .config-toggle {{
            font-size: 18px;
            transition: transform 0.3s;
        }}
        .config-toggle.open {{
            transform: rotate(180deg);
        }}
        .config-body {{
            padding: 20px;
            display: none;
            border-top: 1px solid #ddd;
        }}
        .config-body.open {{
            display: block;
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .config-group {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 6px;
        }}
        .config-group h4 {{
            margin: 0 0 12px 0;
            color: #555;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .config-field {{
            margin-bottom: 12px;
        }}
        .config-field:last-child {{
            margin-bottom: 0;
        }}
        .config-field label {{
            display: block;
            font-size: 12px;
            color: #666;
            margin-bottom: 4px;
        }}
        .config-field input {{
            width: 100%;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
            box-sizing: border-box;
        }}
        .config-field input:focus {{
            outline: none;
            border-color: #4CAF50;
        }}
        .config-field .value-display {{
            font-size: 11px;
            color: #888;
            margin-top: 2px;
        }}
        .config-actions {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
        }}
        .config-actions button {{
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        .btn-apply {{
            background: #4CAF50;
            color: white;
        }}
        .btn-apply:hover {{
            background: #45a049;
        }}
        .btn-reset {{
            background: #f44336;
            color: white;
        }}
        .btn-reset:hover {{
            background: #da190b;
        }}
        .btn-export {{
            background: #2196F3;
            color: white;
        }}
        .btn-export:hover {{
            background: #1976D2;
        }}
        .saved-indicator {{
            color: #4CAF50;
            font-size: 12px;
            margin-left: 10px;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .saved-indicator.show {{
            opacity: 1;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .legend {{
            margin-top: 15px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 4px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            vertical-align: middle;
            margin-right: 5px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Sun Hits Indoor Plant - 3D Simulation</h1>
        <h2 style="color: #666; margin-top: 0;">Date: {date_str}</h2>

        <!-- Configuration Panel -->
        <div class="config-panel">
            <div class="config-header" onclick="toggleConfig()">
                <span>⚙️ Configuration <span class="saved-indicator" id="savedIndicator">✓ Saved</span></span>
                <span class="config-toggle" id="configToggle">▼</span>
            </div>
            <div class="config-body" id="configBody">
                <div class="config-grid">
                    <div class="config-group">
                        <h4>Plant Position</h4>
                        <div class="config-field">
                            <label>Center X (meters)</label>
                            <input type="number" id="plantCenterX" step="0.5" min="0" max="20">
                            <div class="value-display">Distance along Wall 1</div>
                        </div>
                        <div class="config-field">
                            <label>Center Y (meters)</label>
                            <input type="number" id="plantCenterY" step="0.5" min="0" max="20">
                            <div class="value-display">Distance along Wall 2</div>
                        </div>
                    </div>
                    <div class="config-group">
                        <h4>Plant Size</h4>
                        <div class="config-field">
                            <label>Radius (meters)</label>
                            <input type="number" id="plantRadius" step="0.05" min="0.1" max="2">
                        </div>
                        <div class="config-field">
                            <label>Height (z_max, meters)</label>
                            <input type="number" id="plantZMax" step="0.1" min="0.1" max="5">
                        </div>
                    </div>
                    <div class="config-group">
                        <h4>Wall Properties</h4>
                        <div class="config-field">
                            <label>Wall Thickness (meters)</label>
                            <input type="number" id="wallThickness" step="0.05" min="0" max="1">
                            <div class="value-display">0 = thin plane model</div>
                        </div>
                    </div>
                    <div class="config-group">
                        <h4>Simulation</h4>
                        <div class="config-field">
                            <label>Angular Samples</label>
                            <input type="number" id="sampleAngular" step="1" min="4" max="16">
                        </div>
                        <div class="config-field">
                            <label>Vertical Samples</label>
                            <input type="number" id="sampleVertical" step="1" min="2" max="10">
                        </div>
                    </div>
                </div>
                <div class="config-actions">
                    <button class="btn-apply" onclick="applyConfig()">Apply Changes</button>
                    <button class="btn-reset" onclick="resetConfig()">Reset to Default</button>
                    <button class="btn-export" onclick="exportConfig()">Export JSON</button>
                </div>
            </div>
        </div>

        <div class="controls">
            <div class="time-display">Time: <span id="currentTime">--:--</span></div>
            <div id="statusDisplay" class="status miss">Loading...</div>
            <div class="info">
                <span id="sunInfo">Sun: Az --°, El --°</span>
                <span id="windowInfo" style="margin-left: 20px;"></span>
                <span id="hitPointsInfo" style="margin-left: 20px;"></span>
            </div>

            <div class="slider-container">
                <input type="range" id="timeSlider" min="0" max="{len(results) - 1}" value="0">
            </div>

            <div class="legend">
                <span class="legend-item"><span class="legend-color" style="background: #90EE90;"></span> Wall 1 windows</span>
                <span class="legend-item"><span class="legend-color" style="background: #87CEEB;"></span> Wall 2 windows</span>
                <span class="legend-item"><span class="legend-color" style="background: #FFC832;"></span> Window receiving sun</span>
                <span class="legend-item"><span class="legend-color" style="background: #228B22;"></span> Plant</span>
                <span class="legend-item"><span class="legend-color" style="background: #FFD700;"></span> Sun rays through window</span>
                <span class="legend-item"><span class="legend-color" style="background: #FF4500;"></span> Ray hitting plant</span>
            </div>
        </div>

        <div class="plot-container">
            <div id="plot3d" style="width: 100%; height: 700px;"></div>
        </div>
    </div>

    <script>
        // Original config from file (immutable)
        const originalConfig = {config_json};
        const sunPositions = {results_json};

        // Active config (mutable, from localStorage or original)
        let config = JSON.parse(JSON.stringify(originalConfig));
        let results = [];  // Will be recalculated

        // Simulation settings
        let simSettings = {{
            sampleAngular: 8,
            sampleVertical: 3
        }};

        let currentIndex = 0;
        const STORAGE_KEY = 'sunPlantSimulator_config';

        // Room offset - moves room away from origin so sun can be rendered in all directions
        const ROOM_OFFSET_X = 30;
        const ROOM_OFFSET_Y = 30;

        // ========== CONFIG PANEL FUNCTIONS ==========

        function toggleConfig() {{
            const body = document.getElementById('configBody');
            const toggle = document.getElementById('configToggle');
            body.classList.toggle('open');
            toggle.classList.toggle('open');
        }}

        function loadConfigFromStorage() {{
            try {{
                const saved = localStorage.getItem(STORAGE_KEY);
                if (saved) {{
                    const parsed = JSON.parse(saved);
                    // Merge saved config into active config
                    if (parsed.plant) {{
                        config.plant = {{ ...config.plant, ...parsed.plant }};
                    }}
                    if (parsed.wallThickness !== undefined) {{
                        config.windows.forEach(w => w.wall_thickness = parsed.wallThickness);
                    }}
                    if (parsed.simulation) {{
                        simSettings = {{ ...simSettings, ...parsed.simulation }};
                    }}
                    console.log('Loaded config from localStorage:', parsed);
                }}
            }} catch (e) {{
                console.warn('Failed to load config from localStorage:', e);
            }}
        }}

        function saveConfigToStorage() {{
            try {{
                const toSave = {{
                    plant: {{
                        center_x: config.plant.center_x,
                        center_y: config.plant.center_y,
                        radius: config.plant.radius,
                        z_max: config.plant.z_max
                    }},
                    wallThickness: config.windows[0]?.wall_thickness || 0,
                    simulation: simSettings
                }};
                localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));

                // Show saved indicator
                const indicator = document.getElementById('savedIndicator');
                indicator.classList.add('show');
                setTimeout(() => indicator.classList.remove('show'), 2000);
            }} catch (e) {{
                console.warn('Failed to save config to localStorage:', e);
            }}
        }}

        function populateConfigFields() {{
            document.getElementById('plantCenterX').value = config.plant.center_x;
            document.getElementById('plantCenterY').value = config.plant.center_y;
            document.getElementById('plantRadius').value = config.plant.radius;
            document.getElementById('plantZMax').value = config.plant.z_max;
            document.getElementById('wallThickness').value = config.windows[0]?.wall_thickness || 0;
            document.getElementById('sampleAngular').value = simSettings.sampleAngular;
            document.getElementById('sampleVertical').value = simSettings.sampleVertical;
        }}

        function applyConfig() {{
            // Read values from inputs
            config.plant.center_x = parseFloat(document.getElementById('plantCenterX').value);
            config.plant.center_y = parseFloat(document.getElementById('plantCenterY').value);
            config.plant.radius = parseFloat(document.getElementById('plantRadius').value);
            config.plant.z_max = parseFloat(document.getElementById('plantZMax').value);

            const thickness = parseFloat(document.getElementById('wallThickness').value);
            config.windows.forEach(w => w.wall_thickness = thickness);

            simSettings.sampleAngular = parseInt(document.getElementById('sampleAngular').value);
            simSettings.sampleVertical = parseInt(document.getElementById('sampleVertical').value);

            // Save and recalculate
            saveConfigToStorage();
            recalculateAllResults();
            updatePlot(currentIndex);
        }}

        function resetConfig() {{
            if (confirm('Reset all settings to default values?')) {{
                localStorage.removeItem(STORAGE_KEY);
                config = JSON.parse(JSON.stringify(originalConfig));
                simSettings = {{ sampleAngular: 8, sampleVertical: 3 }};
                populateConfigFields();
                recalculateAllResults();
                updatePlot(currentIndex);
            }}
        }}

        function exportConfig() {{
            const exportData = {{
                plant: config.plant,
                windows: config.windows.map(w => ({{
                    id: w.id,
                    center: w.center,
                    width: w.width,
                    height: w.height,
                    wall_thickness: w.wall_thickness
                }})),
                simulation: simSettings
            }};
            const blob = new Blob([JSON.stringify(exportData, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'sun_plant_config.json';
            a.click();
            URL.revokeObjectURL(url);
        }}

        // ========== RAY CASTING (JavaScript implementation) ==========

        function generatePlantSamplePoints() {{
            const points = [];
            const plant = config.plant;
            const nAngular = simSettings.sampleAngular;
            const nVertical = simSettings.sampleVertical;

            for (let vi = 0; vi < nVertical; vi++) {{
                const z = plant.z_min + (plant.z_max - plant.z_min) * (vi + 0.5) / nVertical;
                for (let ai = 0; ai < nAngular; ai++) {{
                    const angle = (2 * Math.PI * ai) / nAngular;
                    const x = plant.center_x + plant.radius * Math.cos(angle);
                    const y = plant.center_y + plant.radius * Math.sin(angle);
                    points.push([x, y, z]);
                }}
            }}
            return points;
        }}

        function sunDirectionFromAngles(azimuthDeg, elevationDeg) {{
            // In the simplified coordinate system, wall 1 is at y=0 along X-axis.
            // The real-world wall 1 has outward normal at wall1_normal_azimuth.
            // We need to rotate the sun azimuth from ENU (real-world) into simplified coords.
            // The rotation angle is: wall1_normal_azimuth - 180° (inward normal direction).
            const wall1NormalAz = config.walls?.find(w => w.id === 'wall_1')?.normal_azimuth || 210;
            const rotation = wall1NormalAz - 180;  // 210 - 180 = 30° for default config
            const rotatedAzimuth = azimuthDeg - rotation;

            const azRad = rotatedAzimuth * Math.PI / 180;
            const elRad = elevationDeg * Math.PI / 180;
            return [
                Math.sin(azRad) * Math.cos(elRad),
                Math.cos(azRad) * Math.cos(elRad),
                Math.sin(elRad)
            ];
        }}

        function rayIntersectsWindowTunnel(origin, direction, window) {{
            // Check sun is on correct side of wall
            const azRad = window.wall_normal_azimuth * Math.PI / 180;
            const wallNormal = [Math.sin(azRad), Math.cos(azRad), 0];
            const sunSideCheck = direction[0]*wallNormal[0] + direction[1]*wallNormal[1];
            if (sunSideCheck <= 0) return false;

            // Determine plane axis (wall 1 at y=0, wall 2 at x=0)
            const isWall1 = Math.abs(window.center[1]) < Math.abs(window.center[0]);
            const planeAxis = isWall1 ? 1 : 0;
            const innerCoord = window.center[planeAxis];
            const thickness = window.wall_thickness || 0;
            const outerCoord = innerCoord - thickness;

            // Helper to check intersection with axis-aligned plane
            function checkPlaneIntersection(planeCoord) {{
                if (Math.abs(direction[planeAxis]) < 1e-10) return null;
                const t = (planeCoord - origin[planeAxis]) / direction[planeAxis];
                if (t < 0) return null;

                const intersection = [
                    origin[0] + t * direction[0],
                    origin[1] + t * direction[1],
                    origin[2] + t * direction[2]
                ];

                // Check bounds
                const localH = isWall1
                    ? intersection[0] - window.center[0]
                    : intersection[1] - window.center[1];
                const localV = intersection[2] - window.center[2];

                const withinH = Math.abs(localH) <= window.width / 2 + 1e-6;
                const withinV = Math.abs(localV) <= window.height / 2 + 1e-6;

                return withinH && withinV ? intersection : null;
            }}

            // Check inner plane
            const innerHit = checkPlaneIntersection(innerCoord);
            if (!innerHit) return false;

            // If no thickness, single plane is enough
            if (thickness <= 0) return true;

            // Check outer plane for tunnel model
            const outerHit = checkPlaneIntersection(outerCoord);
            return outerHit !== null;
        }}

        function checkSunHitsPlant(azimuthDeg, elevationDeg) {{
            if (elevationDeg <= 0) {{
                return {{ is_hit: false, reason: 'sun_below_horizon', sun_direction: null, window_id: null, hit_points: [] }};
            }}

            const sunDir = sunDirectionFromAngles(azimuthDeg, elevationDeg);
            const samplePoints = generatePlantSamplePoints();
            const hitPoints = [];
            let hitWindowId = null;

            for (const pt of samplePoints) {{
                for (const window of config.windows) {{
                    if (rayIntersectsWindowTunnel(pt, sunDir, window)) {{
                        hitPoints.push(pt);
                        if (!hitWindowId) hitWindowId = window.id;
                        break;
                    }}
                }}
            }}

            return {{
                is_hit: hitPoints.length > 0,
                window_id: hitWindowId,
                hit_points: hitPoints,
                sun_direction: sunDir,
                reason: hitPoints.length === 0 ? 'no_window_path' : null
            }};
        }}

        function recalculateAllResults() {{
            results = sunPositions.map(sp => {{
                const hit = checkSunHitsPlant(sp.azimuth, sp.elevation);
                return {{
                    timestamp: sp.timestamp,
                    azimuth: sp.azimuth,
                    elevation: sp.elevation,
                    ...hit
                }};
            }});
        }}

        // Check if sun shines into a window based on wall normal azimuth
        // Sun shines through window if sun direction is opposite to outward normal
        function windowReceivesSun(wallNormalAzimuth, sunDir) {{
            if (!sunDir) return false;

            // Convert wall normal azimuth to direction vector
            const azRad = wallNormalAzimuth * Math.PI / 180;
            const outwardNormal = [Math.sin(azRad), Math.cos(azRad), 0];

            // Inward normal is opposite of outward
            const inwardNormal = [-outwardNormal[0], -outwardNormal[1], 0];

            // Sun shines through if sun direction has component INTO the room
            // (dot product of sun direction and inward normal > 0)
            // But sun_direction points TOWARD sun, so light comes FROM opposite direction
            // Light direction = -sunDir, so we check if -sunDir dot inwardNormal > 0
            // Which is equivalent to: sunDir dot inwardNormal < 0
            // Or: sunDir dot outwardNormal > 0 (sun is on the outside)
            const dot = sunDir[0] * outwardNormal[0] + sunDir[1] * outwardNormal[1];
            return dot > 0.1;  // Sun is in direction of outward normal (outside the room)
        }}

        function createPlotData(index) {{
            const result = results[index];
            const traces = [];
            const sunDir = result.sun_direction;
            const wallThickness = config.windows[0]?.wall_thickness || 0;
            const wallHeight = 6;
            const wallLength = 15;

            // Get wall info
            const wall1 = config.walls?.find(w => w.id === 'wall_1') || {{ normal_azimuth: 210, thickness: 0.3 }};
            const wall2 = config.walls?.find(w => w.id === 'wall_2') || {{ normal_azimuth: 300, thickness: 0.3 }};

            // ========== COMPASS (ENU: X=East, Y=North) ==========
            const compassCenter = [ROOM_OFFSET_X - 3, ROOM_OFFSET_Y - 3, 0.1];
            const compassLen = 2;
            // North arrow (Y+ direction)
            traces.push({{
                type: 'scatter3d',
                mode: 'lines+text',
                x: [compassCenter[0], compassCenter[0]],
                y: [compassCenter[1], compassCenter[1] + compassLen],
                z: [compassCenter[2], compassCenter[2]],
                line: {{ color: 'red', width: 5 }},
                text: ['', 'N'],
                textposition: 'top center',
                textfont: {{ size: 16, color: 'red' }},
                name: 'North',
                showlegend: false,
            }});
            // East arrow (X+ direction)
            traces.push({{
                type: 'scatter3d',
                mode: 'lines+text',
                x: [compassCenter[0], compassCenter[0] + compassLen],
                y: [compassCenter[1], compassCenter[1]],
                z: [compassCenter[2], compassCenter[2]],
                line: {{ color: 'blue', width: 5 }},
                text: ['', 'E'],
                textposition: 'middle right',
                textfont: {{ size: 16, color: 'blue' }},
                name: 'East',
                showlegend: false,
            }});

            // ========== WALL GEOMETRY (simplified axis-aligned) ==========
            // In the simplified coordinate system:
            // - Wall 1 runs along the X axis at y=0 (normal points -Y, azimuth 180° in simplified)
            // - Wall 2 runs along the Y axis at x=0 (normal points -X, azimuth 270° in simplified)
            // The actual wall azimuths (210°, 300°) define the real-world orientation
            // but the coordinates use simplified axis-aligned geometry

            // Simplified wall normals (axis-aligned)
            const wall1Normal = [0, -1];  // Wall 1 at y=0, normal points -Y
            const wall2Normal = [-1, 0];  // Wall 2 at x=0, normal points -X

            // Wall directions (perpendicular to normal, along the wall)
            const wall1Dir = [1, 0];   // Wall 1 runs along +X
            const wall2Dir = [0, 1];   // Wall 2 runs along +Y

            // Corner is at origin in simplified coords, offset for visualization
            const plant = config.plant;
            const plantVizX = ROOM_OFFSET_X;
            const plantVizY = ROOM_OFFSET_Y;
            // In simplified coords, plant.center_x is distance from wall_2 (x=0)
            // and plant.center_y is distance from wall_1 (y=0)
            // So corner in viz coords = plant viz position - plant simplified coords
            const cornerX = plantVizX - plant.center_x;
            const cornerY = plantVizY - plant.center_y;

            // ========== WALL 1 (at y=0, runs along X) ==========
            // Wall runs from corner along wall1Dir
            const w1InnerStart = [cornerX, cornerY];
            const w1InnerEnd = [cornerX + wallLength * wall1Dir[0], cornerY + wallLength * wall1Dir[1]];
            const w1OuterStart = [cornerX - wallThickness * wall1Normal[0], cornerY - wallThickness * wall1Normal[1]];
            const w1OuterEnd = [w1InnerEnd[0] - wallThickness * wall1Normal[0], w1InnerEnd[1] - wallThickness * wall1Normal[1]];

            // Inner surface
            traces.push({{
                type: 'mesh3d',
                x: [w1InnerStart[0], w1InnerEnd[0], w1InnerEnd[0], w1InnerStart[0]],
                y: [w1InnerStart[1], w1InnerEnd[1], w1InnerEnd[1], w1InnerStart[1]],
                z: [0, 0, wallHeight, wallHeight],
                i: [0, 0],
                j: [1, 2],
                k: [2, 3],
                color: 'rgba(180, 180, 180, 0.3)',
                name: 'Wall 1 Inner',
                showlegend: false,
                hoverinfo: 'name',
            }});

            // Outer surface
            if (wallThickness > 0) {{
                traces.push({{
                    type: 'mesh3d',
                    x: [w1OuterStart[0], w1OuterEnd[0], w1OuterEnd[0], w1OuterStart[0]],
                    y: [w1OuterStart[1], w1OuterEnd[1], w1OuterEnd[1], w1OuterStart[1]],
                    z: [0, 0, wallHeight, wallHeight],
                    i: [0, 0],
                    j: [1, 2],
                    k: [2, 3],
                    color: 'rgba(150, 150, 150, 0.2)',
                    name: 'Wall 1 Outer',
                    showlegend: false,
                    hoverinfo: 'name',
                }});
            }}

            // Wall 1 frame
            traces.push({{
                type: 'scatter3d',
                mode: 'lines',
                x: [w1InnerStart[0], w1InnerEnd[0], w1InnerEnd[0], w1InnerStart[0], w1InnerStart[0]],
                y: [w1InnerStart[1], w1InnerEnd[1], w1InnerEnd[1], w1InnerStart[1], w1InnerStart[1]],
                z: [0, 0, wallHeight, wallHeight, 0],
                line: {{ color: '#2E7D32', width: 4 }},
                name: `Wall 1 (azimuth ${{wall1.normal_azimuth}}°)`,
                showlegend: true,
            }});

            // ========== WALL 2 (normal at azimuth ${{wall2.normal_azimuth}}°) ==========
            // Wall runs from corner along wall2Dir
            const w2InnerStart = [cornerX, cornerY];
            const w2InnerEnd = [cornerX + wallLength * wall2Dir[0], cornerY + wallLength * wall2Dir[1]];
            const w2OuterStart = [cornerX - wallThickness * wall2Normal[0], cornerY - wallThickness * wall2Normal[1]];
            const w2OuterEnd = [w2InnerEnd[0] - wallThickness * wall2Normal[0], w2InnerEnd[1] - wallThickness * wall2Normal[1]];

            // Inner surface
            traces.push({{
                type: 'mesh3d',
                x: [w2InnerStart[0], w2InnerEnd[0], w2InnerEnd[0], w2InnerStart[0]],
                y: [w2InnerStart[1], w2InnerEnd[1], w2InnerEnd[1], w2InnerStart[1]],
                z: [0, 0, wallHeight, wallHeight],
                i: [0, 0],
                j: [1, 2],
                k: [2, 3],
                color: 'rgba(180, 180, 180, 0.3)',
                name: 'Wall 2 Inner',
                showlegend: false,
                hoverinfo: 'name',
            }});

            // Outer surface
            if (wallThickness > 0) {{
                traces.push({{
                    type: 'mesh3d',
                    x: [w2OuterStart[0], w2OuterEnd[0], w2OuterEnd[0], w2OuterStart[0]],
                    y: [w2OuterStart[1], w2OuterEnd[1], w2OuterEnd[1], w2OuterStart[1]],
                    z: [0, 0, wallHeight, wallHeight],
                    i: [0, 0],
                    j: [1, 2],
                    k: [2, 3],
                    color: 'rgba(150, 150, 150, 0.2)',
                    name: 'Wall 2 Outer',
                    showlegend: false,
                    hoverinfo: 'name',
                }});
            }}

            // Wall 2 frame
            traces.push({{
                type: 'scatter3d',
                mode: 'lines',
                x: [w2InnerStart[0], w2InnerEnd[0], w2InnerEnd[0], w2InnerStart[0], w2InnerStart[0]],
                y: [w2InnerStart[1], w2InnerEnd[1], w2InnerEnd[1], w2InnerStart[1], w2InnerStart[1]],
                z: [0, 0, wallHeight, wallHeight, 0],
                line: {{ color: '#1565C0', width: 4 }},
                name: `Wall 2 (azimuth ${{wall2.normal_azimuth}}°)`,
                showlegend: true,
            }});

            // ========== FLOOR (triangular, bounded by rotated walls) ==========
            // Floor is the area between the two walls and extending inward
            const floorCorners = [
                [cornerX, cornerY],  // Corner
                [w1InnerEnd[0], w1InnerEnd[1]],  // End of wall 1
                [cornerX + wallLength * wall1Dir[0] + wallLength * wall2Dir[0],
                 cornerY + wallLength * wall1Dir[1] + wallLength * wall2Dir[1]],  // Far corner
                [w2InnerEnd[0], w2InnerEnd[1]],  // End of wall 2
            ];
            traces.push({{
                type: 'mesh3d',
                x: floorCorners.map(c => c[0]),
                y: floorCorners.map(c => c[1]),
                z: [0, 0, 0, 0],
                i: [0, 0],
                j: [1, 2],
                k: [2, 3],
                color: 'rgba(200, 180, 160, 0.3)',
                name: 'Floor',
                showlegend: false,
                hoverinfo: 'name',
            }});

            // Add windows with sun exposure coloring (positioned on rotated walls)
            config.windows.forEach((w, i) => {{
                const isWall1 = w.wall_id === 'wall_1' || w.id.startsWith('window_1');
                const receivesSun = windowReceivesSun(w.wall_normal_azimuth, sunDir);
                const thickness = w.wall_thickness || 0;

                // Colors: bright yellow/orange if receiving sun, otherwise default
                let color, frameColor, outerColor;
                if (receivesSun) {{
                    color = 'rgba(255, 200, 50, 0.8)';  // Bright yellow - sun hitting
                    outerColor = 'rgba(255, 220, 100, 0.6)';
                    frameColor = '#FF8C00';  // Dark orange frame
                }} else {{
                    color = isWall1 ? 'rgba(144, 238, 144, 0.5)' : 'rgba(135, 206, 235, 0.5)';
                    outerColor = isWall1 ? 'rgba(144, 238, 144, 0.3)' : 'rgba(135, 206, 235, 0.3)';
                    frameColor = isWall1 ? '#228B22' : '#4169E1';
                }}

                // Window position along wall (from config x_position or y_position)
                const wallPos = isWall1 ? (w.center[0] || 0) : (w.center[1] || 0);
                const wallDir = isWall1 ? wall1Dir : wall2Dir;
                const wallNormal = isWall1 ? wall1Normal : wall2Normal;

                // Window center on wall (in world coordinates)
                const windowCenterX = cornerX + wallPos * wallDir[0];
                const windowCenterY = cornerY + wallPos * wallDir[1];

                // Window corners (perpendicular to wall = along wallDir, vertical = Z)
                const halfW = w.width / 2;
                const halfH = w.height / 2;
                const wz = w.center[2];

                let innerCorners = [
                    [windowCenterX - halfW * wallDir[0], windowCenterY - halfW * wallDir[1], wz - halfH],
                    [windowCenterX + halfW * wallDir[0], windowCenterY + halfW * wallDir[1], wz - halfH],
                    [windowCenterX + halfW * wallDir[0], windowCenterY + halfW * wallDir[1], wz + halfH],
                    [windowCenterX - halfW * wallDir[0], windowCenterY - halfW * wallDir[1], wz + halfH],
                ];

                // Outer corners (offset by wall thickness in outward normal direction)
                let outerCorners = innerCorners.map(c => [
                    c[0] - thickness * wallNormal[0],
                    c[1] - thickness * wallNormal[1],
                    c[2]
                ]);

                // Inner window mesh
                traces.push({{
                    type: 'mesh3d',
                    x: innerCorners.map(c => c[0]),
                    y: innerCorners.map(c => c[1]),
                    z: innerCorners.map(c => c[2]),
                    i: [0, 0],
                    j: [1, 2],
                    k: [2, 3],
                    color: color,
                    name: w.id + ' (inner)',
                    showlegend: false,
                    hoverinfo: 'name',
                }});

                // Inner window frame
                const innerFrameX = [...innerCorners.map(c => c[0]), innerCorners[0][0]];
                const innerFrameY = [...innerCorners.map(c => c[1]), innerCorners[0][1]];
                const innerFrameZ = [...innerCorners.map(c => c[2]), innerCorners[0][2]];
                traces.push({{
                    type: 'scatter3d',
                    mode: 'lines',
                    x: innerFrameX,
                    y: innerFrameY,
                    z: innerFrameZ,
                    line: {{ color: frameColor, width: 3 }},
                    name: w.id,
                    showlegend: i === 0,
                }});

                // Outer window (only if wall has thickness)
                if (thickness > 0) {{
                    // Outer window mesh
                    traces.push({{
                        type: 'mesh3d',
                        x: outerCorners.map(c => c[0]),
                        y: outerCorners.map(c => c[1]),
                        z: outerCorners.map(c => c[2]),
                        i: [0, 0],
                        j: [1, 2],
                        k: [2, 3],
                        color: outerColor,
                        name: w.id + ' (outer)',
                        showlegend: false,
                        hoverinfo: 'name',
                    }});

                    // Outer window frame
                    const outerFrameX = [...outerCorners.map(c => c[0]), outerCorners[0][0]];
                    const outerFrameY = [...outerCorners.map(c => c[1]), outerCorners[0][1]];
                    const outerFrameZ = [...outerCorners.map(c => c[2]), outerCorners[0][2]];
                    traces.push({{
                        type: 'scatter3d',
                        mode: 'lines',
                        x: outerFrameX,
                        y: outerFrameY,
                        z: outerFrameZ,
                        line: {{ color: frameColor, width: 2, dash: 'dot' }},
                        name: w.id + ' outer frame',
                        showlegend: false,
                    }});

                    // Tunnel edges (connect inner to outer corners)
                    for (let ci = 0; ci < 4; ci++) {{
                        traces.push({{
                            type: 'scatter3d',
                            mode: 'lines',
                            x: [innerCorners[ci][0], outerCorners[ci][0]],
                            y: [innerCorners[ci][1], outerCorners[ci][1]],
                            z: [innerCorners[ci][2], outerCorners[ci][2]],
                            line: {{ color: frameColor, width: 1 }},
                            showlegend: false,
                            hoverinfo: 'skip',
                        }});
                    }}
                }}
            }});

            // Add plant cylinder
            // Plant is positioned at the room offset center (plantVizX, plantVizY)
            // This was calculated earlier: corner + plant.center = plantViz position
            const plantX = plantVizX;
            const plantY = plantVizY;
            const nPts = 20;
            const theta = Array.from({{length: nPts}}, (_, i) => 2 * Math.PI * i / nPts);

            // Bottom circle
            const xBottom = theta.map(t => plantX + plant.radius * Math.cos(t));
            const yBottom = theta.map(t => plantY + plant.radius * Math.sin(t));
            const zBottom = theta.map(() => plant.z_min);

            // Top circle
            const xTop = theta.map(t => plantX + plant.radius * Math.cos(t));
            const yTop = theta.map(t => plantY + plant.radius * Math.sin(t));
            const zTop = theta.map(() => plant.z_max);

            // Combine for mesh
            const allX = [...xBottom, ...xTop];
            const allY = [...yBottom, ...yTop];
            const allZ = [...zBottom, ...zTop];

            const iIdx = [], jIdx = [], kIdx = [];
            for (let idx = 0; idx < nPts; idx++) {{
                const next = (idx + 1) % nPts;
                iIdx.push(idx, next);
                jIdx.push(next, next + nPts);
                kIdx.push(idx + nPts, idx + nPts);
            }}

            traces.push({{
                type: 'mesh3d',
                x: allX,
                y: allY,
                z: allZ,
                i: iIdx,
                j: jIdx,
                k: kIdx,
                color: 'rgba(34, 139, 34, 0.8)',
                name: 'Plant',
                showlegend: true,
            }});

            // Helper function to find ray-window intersection (with rotated walls)
            function rayWindowIntersection(origin, direction, window) {{
                // Get actual wall normal from config azimuth
                const azRad = window.wall_normal_azimuth * Math.PI / 180;
                const normal = [Math.sin(azRad), Math.cos(azRad), 0];

                const isWall1 = window.id.startsWith('window_1');
                const wallDir = isWall1 ? wall1Dir : wall2Dir;
                const wallPos = isWall1 ? (window.center[0] || 0) : (window.center[1] || 0);

                // Window center on wall (in world coordinates)
                const windowCenter = [
                    cornerX + wallPos * wallDir[0],
                    cornerY + wallPos * wallDir[1],
                    window.center[2]
                ];

                const denom = direction[0]*normal[0] + direction[1]*normal[1] + direction[2]*normal[2];
                if (Math.abs(denom) < 1e-10) return null;

                const diff = [
                    windowCenter[0] - origin[0],
                    windowCenter[1] - origin[1],
                    windowCenter[2] - origin[2]
                ];
                const t = (diff[0]*normal[0] + diff[1]*normal[1] + diff[2]*normal[2]) / denom;
                if (t < 0) return null;

                return [
                    origin[0] + t * direction[0],
                    origin[1] + t * direction[1],
                    origin[2] + t * direction[2]
                ];
            }}

            // Draw sun rays from far away through windows that receive sunlight
            if (result.sun_direction) {{
                const sunDir = result.sun_direction;
                const sunDist = 80;
                const roomCenter = [ROOM_OFFSET_X + 8, ROOM_OFFSET_Y + 8, 5];

                // Draw rays through each window that receives sun
                config.windows.forEach((w, wIdx) => {{
                    const receivesSunRay = windowReceivesSun(w.wall_normal_azimuth, sunDir);
                    if (!receivesSunRay) return;

                    const isWall1 = w.id.startsWith('window_1');
                    const wallDir = isWall1 ? wall1Dir : wall2Dir;
                    const wallPos = isWall1 ? (w.center[0] || 0) : (w.center[1] || 0);

                    // Window center on rotated wall (in world coordinates)
                    const windowCenter = [
                        cornerX + wallPos * wallDir[0],
                        cornerY + wallPos * wallDir[1],
                        w.center[2]
                    ];

                    // Calculate sun position along the ray direction from window
                    const sunStart = [
                        windowCenter[0] + sunDist * sunDir[0],
                        windowCenter[1] + sunDist * sunDir[1],
                        windowCenter[2] + sunDist * sunDir[2]
                    ];

                    // Draw ray from far away (sun) through window into room
                    // Ray continues 10m past window into room (negative sun direction)
                    const roomEnd = [
                        windowCenter[0] - 10 * sunDir[0],
                        windowCenter[1] - 10 * sunDir[1],
                        windowCenter[2] - 10 * sunDir[2]
                    ];

                    // Only draw up to floor level (z >= 0)
                    if (roomEnd[2] < 0) {{
                        const t = windowCenter[2] / (windowCenter[2] - roomEnd[2]);
                        roomEnd[0] = windowCenter[0] + t * (roomEnd[0] - windowCenter[0]);
                        roomEnd[1] = windowCenter[1] + t * (roomEnd[1] - windowCenter[1]);
                        roomEnd[2] = 0;
                    }}

                    // Draw the incoming ray from sun to window (bright yellow)
                    traces.push({{
                        type: 'scatter3d',
                        mode: 'lines',
                        x: [sunStart[0], windowCenter[0]],
                        y: [sunStart[1], windowCenter[1]],
                        z: [sunStart[2], windowCenter[2]],
                        line: {{ color: 'rgba(255, 215, 0, 0.6)', width: 3 }},
                        name: wIdx === 0 ? 'Sun ray (incoming)' : '',
                        showlegend: wIdx === 0,
                    }});

                    // Draw ray passing through window into room (brighter)
                    traces.push({{
                        type: 'scatter3d',
                        mode: 'lines',
                        x: [windowCenter[0], roomEnd[0]],
                        y: [windowCenter[1], roomEnd[1]],
                        z: [windowCenter[2], roomEnd[2]],
                        line: {{ color: '#FFD700', width: 4 }},
                        name: wIdx === 0 ? 'Sunlight in room' : '',
                        showlegend: wIdx === 0,
                    }});

                    // Mark where light enters window
                    traces.push({{
                        type: 'scatter3d',
                        mode: 'markers',
                        x: [windowCenter[0]],
                        y: [windowCenter[1]],
                        z: [windowCenter[2]],
                        marker: {{ size: 6, color: '#FF6600', symbol: 'circle' }},
                        name: '',
                        showlegend: false,
                    }});
                }});

                // Also draw rays to plant hit points if plant is hit
                if (result.is_hit && result.hit_points) {{
                    const hitWindow = config.windows.find(w => w.id === result.window_id);

                    result.hit_points.forEach((pt, i) => {{
                        if (i < 3) {{ // Limit rays shown
                            // Convert hit point from ENU to visualization coordinates
                            const ptOffset = [pt[0] + cornerX, pt[1] + cornerY, pt[2]];

                            // Find window intersection for this ray
                            let windowPt = null;
                            if (hitWindow) {{
                                windowPt = rayWindowIntersection(ptOffset, sunDir, hitWindow);
                            }}

                            if (windowPt) {{
                                // Draw ray from window to plant (showing light hitting plant)
                                traces.push({{
                                    type: 'scatter3d',
                                    mode: 'lines',
                                    x: [windowPt[0], ptOffset[0]],
                                    y: [windowPt[1], ptOffset[1]],
                                    z: [windowPt[2], ptOffset[2]],
                                    line: {{ color: '#FF4500', width: 5 }},
                                    name: i === 0 ? 'Ray hitting plant' : '',
                                    showlegend: i === 0,
                                }});

                                // Mark hit point on plant
                                traces.push({{
                                    type: 'scatter3d',
                                    mode: 'markers',
                                    x: [ptOffset[0]],
                                    y: [ptOffset[1]],
                                    z: [ptOffset[2]],
                                    marker: {{ size: 8, color: '#FF0000', symbol: 'diamond' }},
                                    name: i === 0 ? 'Plant hit' : '',
                                    showlegend: i === 0,
                                }});
                            }}
                        }}
                    }});
                }}
            }}

            // Add sun indicator - far away so rays appear parallel (from "infinity")
            if (result.sun_direction) {{
                const sunDir = result.sun_direction;
                // Sun very far away - rays will appear nearly parallel
                const sunDist = 80;
                const roomCenter = [ROOM_OFFSET_X + 8, ROOM_OFFSET_Y + 8, 5];
                const sunPos = [
                    roomCenter[0] + sunDist * sunDir[0],
                    roomCenter[1] + sunDist * sunDir[1],
                    Math.max(5, sunDist * sunDir[2]),
                ];

                // Sun marker (larger since it's further away)
                traces.push({{
                    type: 'scatter3d',
                    mode: 'markers',
                    x: [sunPos[0]],
                    y: [sunPos[1]],
                    z: [sunPos[2]],
                    marker: {{ size: 25, color: '#FFD700', symbol: 'circle',
                              line: {{ color: '#FFA500', width: 3 }} }},
                    name: `Sun (El: ${{result.elevation.toFixed(0)}}°)`,
                    showlegend: true,
                }});

                // Draw line from room center toward sun to show direction
                traces.push({{
                    type: 'scatter3d',
                    mode: 'lines',
                    x: [roomCenter[0], sunPos[0]],
                    y: [roomCenter[1], sunPos[1]],
                    z: [roomCenter[2], sunPos[2]],
                    line: {{ color: 'rgba(255, 215, 0, 0.4)', width: 3, dash: 'dot' }},
                    name: 'Sun direction',
                    showlegend: false,
                }});
            }}

            return traces;
        }}

        function updatePlot(index) {{
            const result = results[index];
            const traces = createPlotData(index);

            // Update info displays
            document.getElementById('currentTime').textContent = result.timestamp;
            document.getElementById('sunInfo').textContent =
                `Sun: Az ${{result.azimuth.toFixed(0)}}°, El ${{result.elevation.toFixed(0)}}°`;

            const statusEl = document.getElementById('statusDisplay');
            if (result.is_hit) {{
                statusEl.textContent = '☀️ SUNLIGHT HITTING PLANT';
                statusEl.className = 'status hit';
                document.getElementById('windowInfo').textContent = `Through: ${{result.window_id}}`;
                const totalSamples = simSettings.sampleAngular * simSettings.sampleVertical;
                const hitCount = result.hit_points ? result.hit_points.length : 0;
                document.getElementById('hitPointsInfo').textContent = `Hits: ${{hitCount}}/${{totalSamples}} (${{Math.round(100*hitCount/totalSamples)}}%)`;
            }} else {{
                statusEl.textContent = 'No direct sunlight';
                statusEl.className = 'status miss';
                document.getElementById('windowInfo').textContent = '';
                document.getElementById('hitPointsInfo').textContent = '';
            }}

            Plotly.react('plot3d', traces, {{
                scene: {{
                    xaxis: {{ title: 'X (meters)', range: [-30, 120] }},
                    yaxis: {{ title: 'Y (meters)', range: [-30, 120] }},
                    zaxis: {{ title: 'Z (meters)', range: [0, 80] }},
                    aspectmode: 'cube',
                    camera: {{
                        eye: {{ x: 1.2, y: 1.2, z: 0.7 }}
                    }}
                }},
                title: `Sun-Plant Simulation - ${{result.timestamp}}`,
                showlegend: true,
                legend: {{ x: 0.02, y: 0.98 }},
            }});
        }}

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            // Load saved config from localStorage
            loadConfigFromStorage();

            // Populate form fields with current config
            populateConfigFields();

            // Calculate initial results with current config
            recalculateAllResults();

            const slider = document.getElementById('timeSlider');

            slider.addEventListener('input', function() {{
                currentIndex = parseInt(this.value);
                updatePlot(currentIndex);
            }});

            // Auto-apply on Enter key in inputs
            document.querySelectorAll('.config-field input').forEach(input => {{
                input.addEventListener('keypress', function(e) {{
                    if (e.key === 'Enter') applyConfig();
                }});
            }});

            // Initial plot
            updatePlot(0);
        }});
    </script>
</body>
</html>'''

    return html
