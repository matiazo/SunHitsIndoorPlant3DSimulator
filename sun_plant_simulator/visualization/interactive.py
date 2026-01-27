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
        "windows": [
            {
                "id": w.id,
                "center": w.center.tolist(),
                "width": w.width,
                "height": w.height,
                "wall_normal_azimuth": w.wall_normal_azimuth,
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

        <div class="controls">
            <div class="time-display">Time: <span id="currentTime">--:--</span></div>
            <div id="statusDisplay" class="status miss">Loading...</div>
            <div class="info">
                <span id="sunInfo">Sun: Az --°, El --°</span>
                <span id="windowInfo" style="margin-left: 20px;"></span>
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
        const config = {config_json};
        const results = {results_json};

        let currentIndex = 0;

        // Room offset - moves room away from origin so sun can be rendered in all directions
        const ROOM_OFFSET_X = 30;
        const ROOM_OFFSET_Y = 30;

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

            // Add walls as lines to show room boundary
            // Wall 1 along X-axis (at y=ROOM_OFFSET_Y)
            traces.push({{
                type: 'scatter3d',
                mode: 'lines',
                x: [ROOM_OFFSET_X, ROOM_OFFSET_X + 15],
                y: [ROOM_OFFSET_Y, ROOM_OFFSET_Y],
                z: [0, 0],
                line: {{ color: '#888', width: 4 }},
                name: 'Wall 1 (X-axis)',
                showlegend: false,
            }});
            traces.push({{
                type: 'scatter3d',
                mode: 'lines',
                x: [ROOM_OFFSET_X, ROOM_OFFSET_X + 15],
                y: [ROOM_OFFSET_Y, ROOM_OFFSET_Y],
                z: [6, 6],
                line: {{ color: '#888', width: 4 }},
                name: 'Wall 1 top',
                showlegend: false,
            }});

            // Wall 2 along Y-axis (at x=ROOM_OFFSET_X)
            traces.push({{
                type: 'scatter3d',
                mode: 'lines',
                x: [ROOM_OFFSET_X, ROOM_OFFSET_X],
                y: [ROOM_OFFSET_Y, ROOM_OFFSET_Y + 15],
                z: [0, 0],
                line: {{ color: '#888', width: 4 }},
                name: 'Wall 2 (Y-axis)',
                showlegend: false,
            }});
            traces.push({{
                type: 'scatter3d',
                mode: 'lines',
                x: [ROOM_OFFSET_X, ROOM_OFFSET_X],
                y: [ROOM_OFFSET_Y, ROOM_OFFSET_Y + 15],
                z: [6, 6],
                line: {{ color: '#888', width: 4 }},
                name: 'Wall 2 top',
                showlegend: false,
            }});

            // Add windows with sun exposure coloring
            config.windows.forEach((w, i) => {{
                const isWall1 = w.id.startsWith('window_1');
                const receivesSun = windowReceivesSun(w.wall_normal_azimuth, sunDir);

                // Colors: bright yellow/orange if receiving sun, otherwise default
                let color, frameColor;
                if (receivesSun) {{
                    color = 'rgba(255, 200, 50, 0.8)';  // Bright yellow - sun hitting
                    frameColor = '#FF8C00';  // Dark orange frame
                }} else {{
                    color = isWall1 ? 'rgba(144, 238, 144, 0.4)' : 'rgba(135, 206, 235, 0.4)';
                    frameColor = isWall1 ? '#228B22' : '#4169E1';
                }}

                // Window rectangle with offset
                let corners;
                if (isWall1) {{
                    // Wall 1: along X-axis, y=ROOM_OFFSET_Y
                    corners = [
                        [ROOM_OFFSET_X + w.center[0] - w.width/2, ROOM_OFFSET_Y, w.center[2] - w.height/2],
                        [ROOM_OFFSET_X + w.center[0] + w.width/2, ROOM_OFFSET_Y, w.center[2] - w.height/2],
                        [ROOM_OFFSET_X + w.center[0] + w.width/2, ROOM_OFFSET_Y, w.center[2] + w.height/2],
                        [ROOM_OFFSET_X + w.center[0] - w.width/2, ROOM_OFFSET_Y, w.center[2] + w.height/2],
                    ];
                }} else {{
                    // Wall 2: along Y-axis, x=ROOM_OFFSET_X
                    corners = [
                        [ROOM_OFFSET_X, ROOM_OFFSET_Y + w.center[1] - w.width/2, w.center[2] - w.height/2],
                        [ROOM_OFFSET_X, ROOM_OFFSET_Y + w.center[1] + w.width/2, w.center[2] - w.height/2],
                        [ROOM_OFFSET_X, ROOM_OFFSET_Y + w.center[1] + w.width/2, w.center[2] + w.height/2],
                        [ROOM_OFFSET_X, ROOM_OFFSET_Y + w.center[1] - w.width/2, w.center[2] + w.height/2],
                    ];
                }}

                // Window mesh
                traces.push({{
                    type: 'mesh3d',
                    x: corners.map(c => c[0]),
                    y: corners.map(c => c[1]),
                    z: corners.map(c => c[2]),
                    i: [0, 0],
                    j: [1, 2],
                    k: [2, 3],
                    color: color,
                    name: w.id,
                    showlegend: false,
                }});

                // Window frame
                const frameX = [...corners.map(c => c[0]), corners[0][0]];
                const frameY = [...corners.map(c => c[1]), corners[0][1]];
                const frameZ = [...corners.map(c => c[2]), corners[0][2]];
                traces.push({{
                    type: 'scatter3d',
                    mode: 'lines',
                    x: frameX,
                    y: frameY,
                    z: frameZ,
                    line: {{ color: frameColor, width: 3 }},
                    name: w.id + ' frame',
                    showlegend: false,
                }});
            }});

            // Add plant cylinder (with offset)
            const plant = config.plant;
            const plantX = ROOM_OFFSET_X + plant.center_x;
            const plantY = ROOM_OFFSET_Y + plant.center_y;
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

            // Helper function to find ray-window intersection (with offset)
            function rayWindowIntersection(origin, direction, window) {{
                // Get window normal based on wall
                const isWall1 = window.id.startsWith('window_1');
                // Wall 1 along X-axis (y=OFFSET), normal points in -Y direction
                // Wall 2 along Y-axis (x=OFFSET), normal points in -X direction
                const normal = isWall1 ? [0, -1, 0] : [-1, 0, 0];

                // Window center with offset
                const windowCenter = isWall1
                    ? [ROOM_OFFSET_X + window.center[0], ROOM_OFFSET_Y, window.center[2]]
                    : [ROOM_OFFSET_X, ROOM_OFFSET_Y + window.center[1], window.center[2]];

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

                    // Window center with offset
                    const windowCenter = isWall1
                        ? [ROOM_OFFSET_X + w.center[0], ROOM_OFFSET_Y, w.center[2]]
                        : [ROOM_OFFSET_X, ROOM_OFFSET_Y + w.center[1], w.center[2]];

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
                            const ptOffset = [pt[0] + ROOM_OFFSET_X, pt[1] + ROOM_OFFSET_Y, pt[2]];

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
            }} else {{
                statusEl.textContent = 'No direct sunlight';
                statusEl.className = 'status miss';
                document.getElementById('windowInfo').textContent = '';
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
            const slider = document.getElementById('timeSlider');

            slider.addEventListener('input', function() {{
                currentIndex = parseInt(this.value);
                updatePlot(currentIndex);
            }});

            // Initial plot
            updatePlot(0);
        }});
    </script>
</body>
</html>'''

    return html
