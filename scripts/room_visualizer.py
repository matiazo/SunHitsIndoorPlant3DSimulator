#!/usr/bin/env python3
"""Interactive 3D room visualization for the sun-plant simulator.

Generates a self-contained HTML file with a Plotly.js 3D scene showing
the room, walls, windows, plant, sun path, light volumes, and per-window
hit status over time.

Usage:
    python scripts/room_visualizer.py [YYYY-MM-DD]
    Defaults to today if no date is given.
"""

import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sun_hit_detector.core.models import Config  # noqa: E402
from sun_hit_detector.core.geometry import sun_direction_simplified  # noqa: E402
from sun_hit_detector.core.hit_test import (  # noqa: E402
    check_plant_hit_per_window_from_config,
    generate_plant_sample_points,
)
from sun_hit_detector.core.sun_position import generate_sun_data_for_date  # noqa: E402
from sun_hit_detector.core.window_sun import (  # noqa: E402
    check_window_sun_exposure_geometric,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CEILING_HEIGHT = 6.0
WALL1_NORMAL_AZIMUTH = 210.0
MAX_PROJECTION_T = 30.0
SUN_DISTANCE = 12.0
N_CYLINDER_SEGMENTS = 24


def load_config_and_location(config_path: str):
    """Load Config object and raw location dict from JSON."""
    config = Config.from_json_file(config_path)
    with open(config_path, "r") as f:
        raw = json.load(f)
    location = raw["location"]
    walls_raw = raw["walls"]
    wall_lengths = {}
    for w in walls_raw:
        wall_lengths[w["id"]] = w.get("visualization", {}).get("wall_length", 15.0)
    return config, location, wall_lengths


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def box_mesh(x0, x1, y0, y1, z0, z1):
    """Return (verts_x, verts_y, verts_z, i, j, k) for a Mesh3d box."""
    vx = [x0, x1, x1, x0, x0, x1, x1, x0]
    vy = [y0, y0, y1, y1, y0, y0, y1, y1]
    vz = [z0, z0, z0, z0, z1, z1, z1, z1]
    # 12 triangles, 6 faces
    ii = [0, 0, 1, 1, 0, 0, 4, 4, 0, 0, 1, 1]
    jj = [1, 2, 2, 3, 1, 5, 5, 6, 3, 7, 2, 6]
    kk = [2, 3, 3, 0, 5, 4, 6, 7, 7, 4, 6, 5]
    return vx, vy, vz, ii, jj, kk


def cylinder_mesh(cx, cy, z_min, z_max, radius, n_seg=N_CYLINDER_SEGMENTS):
    """Return (vx, vy, vz, ii, jj, kk) for a closed cylinder Mesh3d."""
    vx, vy, vz = [], [], []
    ii, jj, kk = [], [], []

    # Bottom + top ring vertices
    for k_idx in range(2):
        z = z_min if k_idx == 0 else z_max
        for s in range(n_seg):
            angle = 2 * math.pi * s / n_seg
            vx.append(cx + radius * math.cos(angle))
            vy.append(cy + radius * math.sin(angle))
            vz.append(z)
    # Center bottom and top
    bc = len(vx)
    vx.append(cx); vy.append(cy); vz.append(z_min)
    tc = len(vx)
    vx.append(cx); vy.append(cy); vz.append(z_max)

    # Side triangles
    for s in range(n_seg):
        s_next = (s + 1) % n_seg
        b0, b1 = s, s_next
        t0, t1 = n_seg + s, n_seg + s_next
        ii += [b0, b1]
        jj += [b1, t1]
        kk += [t0, t0]

    # Bottom cap
    for s in range(n_seg):
        s_next = (s + 1) % n_seg
        ii.append(bc); jj.append(s_next); kk.append(s)

    # Top cap
    for s in range(n_seg):
        s_next = (s + 1) % n_seg
        ii.append(tc); jj.append(n_seg + s); kk.append(n_seg + s_next)

    return vx, vy, vz, ii, jj, kk


def frustum_mesh(corners_top, corners_bottom):
    """Build Mesh3d data for a frustum from 4 top + 4 bottom vertices.

    corners_top/bottom: list of 4 [x,y,z] each, same winding order.
    """
    vx, vy, vz = [], [], []
    for p in list(corners_top) + list(corners_bottom):
        vx.append(float(p[0]))
        vy.append(float(p[1]))
        vz.append(float(p[2]))
    # Indices: 0-3 = top (window), 4-7 = bottom (floor)
    ii = [0, 0, 0, 0,  4, 4,  0, 1,  1, 2,  2, 3]
    jj = [1, 1, 2, 3,  5, 6,  1, 5,  2, 6,  3, 7]
    kk = [2, 4, 3, 7,  6, 7,  5, 4,  6, 5,  7, 6]
    # Correct: 4 side faces (2 tri each) + top face (2 tri) + bottom face (2 tri)
    ii = [
        0, 0,  # top face
        4, 4,  # bottom face
        0, 0,  # side 0-1
        1, 1,  # side 1-2
        2, 2,  # side 2-3
        3, 3,  # side 3-0
    ]
    jj = [
        1, 2,
        5, 6,
        1, 5,
        2, 6,
        3, 7,
        0, 4,
    ]
    kk = [
        3, 3,
        4, 4,
        5, 4,
        6, 5,
        7, 6,
        4, 7,
    ]
    return vx, vy, vz, ii, jj, kk


def project_corner_to_floor(corner, light_dir, max_t=MAX_PROJECTION_T):
    """Project a window corner along light_dir until z=0."""
    if light_dir[2] >= 0:
        return None
    t = -corner[2] / light_dir[2]
    t = min(t, max_t)
    return corner + t * light_dir


# ---------------------------------------------------------------------------
# Precompute data for all timesteps
# ---------------------------------------------------------------------------

def precompute_time_data(config, sun_data, wall_lengths):
    """Pre-compute all dynamic geometry for every timestep."""
    plant = config.plant
    windows = config.windows
    sample_pts = generate_plant_sample_points(
        plant,
        config.simulation.sample_points_angular,
        config.simulation.sample_points_vertical,
    )

    # Room center for sun indicator
    w1_len = wall_lengths.get("wall_1", 6.5)
    w2_len = wall_lengths.get("wall_2", 10.0)
    room_cx = w2_len / 2
    room_cy = w1_len / 2
    room_cz = CEILING_HEIGHT / 2

    time_steps = []
    for sd in sun_data:
        az = sd["azimuth_deg"]
        el = sd["elevation_deg"]
        ts = sd["timestamp"]

        sun_dir = sun_direction_simplified(az, el, WALL1_NORMAL_AZIMUTH)
        light_dir = -sun_dir

        # Sun sky marker
        sun_x = room_cx + sun_dir[0] * SUN_DISTANCE
        sun_y = room_cy + sun_dir[1] * SUN_DISTANCE
        sun_z = room_cz + sun_dir[2] * SUN_DISTANCE
        if sun_z < 0:
            sun_z = 0

        # Per-window hit test
        hit_map = check_plant_hit_per_window_from_config(az, el, config)

        # Per-window light geometry
        window_volumes = []
        for w in windows:
            is_facing, angle_deg, intensity = check_window_sun_exposure_geometric(w, sun_dir)
            corners = w.get_corners()  # BL, BR, TR, TL viewed from outside

            vol = {
                "window_id": w.id,
                "is_facing": is_facing,
                "plant_hit": hit_map.get(w.id, False),
                "intensity": float(intensity),
                "angle_deg": float(angle_deg),
                "volume_verts": None,
                "floor_patch": None,
            }

            if is_facing and el > 0:
                floor_pts = []
                for c in corners:
                    fp = project_corner_to_floor(c, light_dir)
                    if fp is not None:
                        floor_pts.append(fp.tolist())
                if len(floor_pts) == 4:
                    vol["volume_verts"] = {
                        "top": [c.tolist() for c in corners],
                        "bottom": floor_pts,
                    }
                    vol["floor_patch"] = floor_pts

            window_volumes.append(vol)

        # Hit rays: from plant sample points toward sun (through hitting windows)
        hit_rays = []
        if el > 0:
            for pt in sample_pts:
                from sun_hit_detector.core.ray_casting import ray_intersects_window
                for w in windows:
                    if hit_map.get(w.id, False) and ray_intersects_window(pt, sun_dir, w):
                        # Ray from plant point toward sun, length ~10m
                        end = pt + sun_dir * 8.0
                        hit_rays.append({
                            "start": pt.tolist(),
                            "end": end.tolist(),
                            "window_id": w.id,
                        })
                        break

        step = {
            "timestamp": ts,
            "azimuth": float(az),
            "elevation": float(el),
            "sun_x": float(sun_x),
            "sun_y": float(sun_y),
            "sun_z": float(sun_z),
            "window_volumes": window_volumes,
            "hit_rays": hit_rays,
            "any_hit": any(hit_map.values()),
        }
        time_steps.append(step)

    return time_steps, [p.tolist() for p in sample_pts]


# ---------------------------------------------------------------------------
# Static trace builders (return dicts for JSON embedding)
# ---------------------------------------------------------------------------

def build_floor_trace(wall_lengths):
    w1_len = wall_lengths.get("wall_1", 6.5)
    w2_len = wall_lengths.get("wall_2", 10.0)
    margin = 0.5
    x = [-margin, w2_len + margin, w2_len + margin, -margin]
    y = [-margin, -margin, w1_len + margin, w1_len + margin]
    z = [0, 0, 0, 0]
    return {
        "type": "mesh3d",
        "x": x, "y": y, "z": z,
        "i": [0, 0], "j": [1, 2], "k": [2, 3],
        "color": "#e0e0e0",
        "opacity": 0.3,
        "name": "Floor",
        "hoverinfo": "name",
        "showlegend": False,
    }


def build_wall_traces(config, wall_lengths):
    """Build wall box meshes. Walls are semi-transparent."""
    traces = []
    thickness = 0.30
    for wall_cfg in config.walls:
        wid = wall_cfg.id
        length = wall_lengths.get(wid, 15.0)
        if wid == "wall_1":
            vx, vy, vz, ii, jj, kk = box_mesh(0, length, -thickness, 0, 0, CEILING_HEIGHT)
        else:
            vx, vy, vz, ii, jj, kk = box_mesh(-thickness, 0, 0, length, 0, CEILING_HEIGHT)
        traces.append({
            "type": "mesh3d",
            "x": vx, "y": vy, "z": vz,
            "i": ii, "j": jj, "k": kk,
            "color": "#b0b0b0",
            "opacity": 0.35,
            "name": wid,
            "hoverinfo": "name",
            "showlegend": False,
        })
    return traces


def build_window_traces(config):
    """Window glass (transparent blue) + frame lines."""
    traces = []
    for w in config.windows:
        corners = w.get_corners()
        xs = [float(c[0]) for c in corners]
        ys = [float(c[1]) for c in corners]
        zs = [float(c[2]) for c in corners]

        # Glass quad
        traces.append({
            "type": "mesh3d",
            "x": xs, "y": ys, "z": zs,
            "i": [0, 0], "j": [1, 2], "k": [2, 3],
            "color": "#4fc3f7",
            "opacity": 0.35,
            "name": w.id,
            "hoverinfo": "name",
            "showlegend": False,
        })

        # Frame border
        fx = xs + [xs[0]]
        fy = ys + [ys[0]]
        fz = zs + [zs[0]]
        traces.append({
            "type": "scatter3d",
            "x": fx, "y": fy, "z": fz,
            "mode": "lines",
            "line": {"color": "#1a237e", "width": 4},
            "name": w.id + " frame",
            "hoverinfo": "name",
            "showlegend": False,
        })
    return traces


def build_plant_trace(plant):
    """Green cylinder for the plant."""
    vx, vy, vz, ii, jj, kk = cylinder_mesh(
        plant.center_x, plant.center_y, plant.z_min, plant.z_max, plant.radius
    )
    return {
        "type": "mesh3d",
        "x": vx, "y": vy, "z": vz,
        "i": ii, "j": jj, "k": kk,
        "color": "#4caf50",
        "opacity": 0.85,
        "name": "Plant",
        "hoverinfo": "name",
        "showlegend": False,
    }


def build_sample_points_trace(sample_pts):
    """Small markers on the plant surface."""
    xs = [p[0] for p in sample_pts]
    ys = [p[1] for p in sample_pts]
    zs = [p[2] for p in sample_pts]
    return {
        "type": "scatter3d",
        "x": xs, "y": ys, "z": zs,
        "mode": "markers",
        "marker": {"size": 2.5, "color": "#2e7d32", "opacity": 0.7},
        "name": "Sample points",
        "hoverinfo": "name",
        "showlegend": False,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(static_traces, time_data, sample_pts, window_ids, date_str,
                  wall_lengths, plant):
    """Build self-contained HTML string."""

    # Encode data as JSON for embedding
    static_json = json.dumps(static_traces)
    time_json = json.dumps(time_data)
    window_ids_json = json.dumps(window_ids)

    w1_len = wall_lengths.get("wall_1", 6.5)
    w2_len = wall_lengths.get("wall_2", 10.0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Room Sun Visualization — {date_str}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #eee; }}
#container {{ display: flex; height: 100vh; }}
#plot {{ flex: 1; min-width: 0; }}
#sidebar {{
    width: 320px; min-width: 280px; background: #16213e; padding: 16px;
    overflow-y: auto; display: flex; flex-direction: column; gap: 12px;
    border-left: 2px solid #0f3460;
}}
h2 {{ color: #e94560; font-size: 1.1rem; margin-bottom: 4px; }}
.card {{
    background: #0f3460; border-radius: 8px; padding: 12px;
}}
.card label {{ display: block; font-size: 0.85rem; color: #aaa; margin-bottom: 4px; }}
.card .value {{ font-size: 1.2rem; font-weight: 600; }}
input[type=range] {{ width: 100%; accent-color: #e94560; }}
input[type=date] {{ width: 100%; padding: 4px 8px; border-radius: 4px; border: 1px solid #0f3460; background: #1a1a2e; color: #eee; }}
.btn {{
    padding: 6px 16px; border: none; border-radius: 4px; cursor: pointer;
    font-weight: 600; font-size: 0.9rem;
}}
.btn-play {{ background: #e94560; color: #fff; }}
.btn-play:hover {{ background: #c81e45; }}
.win-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
.win-cell {{
    padding: 6px 8px; border-radius: 6px; font-size: 0.8rem; text-align: center;
}}
.win-hit {{ background: #ff9800; color: #000; font-weight: 700; }}
.win-sun {{ background: #fdd835; color: #000; }}
.win-none {{ background: #263238; color: #78909c; }}
</style>
</head>
<body>
<div id="container">
    <div id="plot"></div>
    <div id="sidebar">
        <h2>☀️ Sun-Plant Visualizer</h2>
        <div class="card">
            <label>Date</label>
            <input type="date" id="dateInput" value="{date_str}">
        </div>
        <div class="card">
            <label>Time</label>
            <div style="display:flex;align-items:center;gap:8px;">
                <input type="range" id="timeSlider" min="0" max="0" value="0" style="flex:1;">
                <button class="btn btn-play" id="playBtn">▶</button>
            </div>
            <div class="value" id="timeLabel" style="margin-top:4px;">--:--</div>
        </div>
        <div class="card">
            <label>Sun Position</label>
            <div id="sunInfo" class="value" style="font-size:0.95rem;">—</div>
        </div>
        <div class="card">
            <label>Plant Status</label>
            <div id="plantStatus" class="value" style="font-size:1rem;">—</div>
        </div>
        <div class="card">
            <label>Window Status</label>
            <div class="win-grid" id="winGrid"></div>
        </div>
        <div class="card" style="font-size:0.75rem;color:#78909c;">
            Room: {w2_len:.1f} × {w1_len:.1f} m &bull; Plant at ({plant.center_x:.1f}, {plant.center_y:.1f})<br>
            Ceiling: {CEILING_HEIGHT:.1f} m &bull; {len(window_ids)} windows
        </div>
    </div>
</div>
<script>
// ---- Embedded data ----
const staticTraces = {static_json};
const timeData = {time_json};
const windowIds = {window_ids_json};
const NUM_WINDOWS = windowIds.length;
const MAX_RAYS = 30;  // pool size for hit-ray traces

// ---- Build initial traces ----
// Static traces come first.
// Then dynamic traces in known order:
//   [S]   = sun marker        (1 trace)
//   [S+1 .. S+N] = light volumes per window (N traces)
//   [S+N+1 .. S+2N] = floor patches per window (N traces)
//   [S+2N+1 .. S+2N+MAX_RAYS] = hit rays (MAX_RAYS traces)
//   [S+2N+MAX_RAYS+1 .. S+2N+MAX_RAYS+N] = window labels (N traces)

const S = staticTraces.length;
const allTraces = [...staticTraces];

// Sun marker
allTraces.push({{
    type: 'scatter3d', x: [0], y: [0], z: [0],
    mode: 'markers',
    marker: {{ size: 14, color: '#fdd835', symbol: 'circle',
              line: {{ color: '#ff6f00', width: 2 }} }},
    name: 'Sun', hoverinfo: 'name', showlegend: false
}});
const sunIdx = S;

// Light volume traces (one per window)
const volStartIdx = S + 1;
for (let i = 0; i < NUM_WINDOWS; i++) {{
    allTraces.push({{
        type: 'mesh3d', x: [0], y: [0], z: [0],
        i: [], j: [], k: [],
        color: '#ffeb3b', opacity: 0.12,
        name: 'Light ' + windowIds[i], hoverinfo: 'name', showlegend: false,
        visible: false
    }});
}}

// Floor patch traces
const floorStartIdx = volStartIdx + NUM_WINDOWS;
for (let i = 0; i < NUM_WINDOWS; i++) {{
    allTraces.push({{
        type: 'mesh3d', x: [0], y: [0], z: [0],
        i: [0, 0], j: [1, 2], k: [2, 3],
        color: '#fff9c4', opacity: 0.4,
        name: 'Floor patch ' + windowIds[i], hoverinfo: 'name', showlegend: false,
        visible: false
    }});
}}

// Hit ray pool
const rayStartIdx = floorStartIdx + NUM_WINDOWS;
for (let i = 0; i < MAX_RAYS; i++) {{
    allTraces.push({{
        type: 'scatter3d', x: [0, 0], y: [0, 0], z: [0, 0],
        mode: 'lines',
        line: {{ color: '#ffab00', width: 2 }},
        name: 'Ray', hoverinfo: 'skip', showlegend: false,
        visible: false, opacity: 0.6
    }});
}}

// Window label traces
const labelStartIdx = rayStartIdx + MAX_RAYS;
for (let i = 0; i < NUM_WINDOWS; i++) {{
    allTraces.push({{
        type: 'scatter3d', x: [0], y: [0], z: [0],
        mode: 'text',
        text: [''],
        textfont: {{ size: 11, color: '#fff' }},
        name: 'Label ' + windowIds[i], hoverinfo: 'skip', showlegend: false
    }});
}}

// ---- Layout ----
const layout = {{
    scene: {{
        xaxis: {{ title: 'X (m)', range: [-1, {w2_len + 2}], backgroundcolor: '#1a1a2e', gridcolor: '#333', color: '#aaa' }},
        yaxis: {{ title: 'Y (m)', range: [-1, {w1_len + 2}], backgroundcolor: '#1a1a2e', gridcolor: '#333', color: '#aaa' }},
        zaxis: {{ title: 'Z (m)', range: [-0.5, {CEILING_HEIGHT + 2}], backgroundcolor: '#1a1a2e', gridcolor: '#333', color: '#aaa' }},
        aspectmode: 'data',
        camera: {{
            eye: {{ x: 1.8, y: -1.5, z: 1.0 }},
            center: {{ x: 0, y: 0, z: -0.15 }}
        }},
        bgcolor: '#1a1a2e'
    }},
    paper_bgcolor: '#1a1a2e',
    margin: {{ l: 0, r: 0, t: 0, b: 0 }},
    showlegend: false
}};

Plotly.newPlot('plot', allTraces, layout, {{ responsive: true }});

// ---- UI Controls ----
const slider = document.getElementById('timeSlider');
const timeLabel = document.getElementById('timeLabel');
const sunInfo = document.getElementById('sunInfo');
const plantStatus = document.getElementById('plantStatus');
const winGrid = document.getElementById('winGrid');
const playBtn = document.getElementById('playBtn');

slider.max = Math.max(0, timeData.length - 1);
if (timeData.length > 0) slider.value = 0;

// Build window grid cells
windowIds.forEach(wid => {{
    const cell = document.createElement('div');
    cell.className = 'win-cell win-none';
    cell.id = 'wc-' + wid;
    cell.textContent = wid.replace('window_', 'W');
    winGrid.appendChild(cell);
}});

function updateTime(idx) {{
    if (idx < 0 || idx >= timeData.length) return;
    const d = timeData[idx];

    // Time label
    timeLabel.textContent = d.timestamp;
    sunInfo.innerHTML = 'Az: ' + d.azimuth.toFixed(1) + '°  El: ' + d.elevation.toFixed(1) + '°';

    // Plant status
    if (d.any_hit) {{
        const hitWins = d.window_volumes.filter(v => v.plant_hit).map(v => v.window_id.replace('window_', 'W'));
        plantStatus.innerHTML = '🌞 <span style="color:#ff9800">HIT</span> via ' + hitWins.join(', ');
    }} else if (d.elevation <= 0) {{
        plantStatus.innerHTML = '🌑 Night';
    }} else {{
        plantStatus.innerHTML = '🌤️ No direct sun on plant';
    }}

    const updates = {{}};
    const indices = [];

    // Sun marker
    indices.push(sunIdx);
    updates[indices.length - 1] = {{
        x: [[d.sun_x]], y: [[d.sun_y]], z: [[d.sun_z]]
    }};

    // We'll batch restyle calls per trace
    const restyles = [];

    // Sun marker restyle
    restyles.push([{{ 'x': [[d.sun_x]], 'y': [[d.sun_y]], 'z': [[d.sun_z]] }}, [sunIdx]]);

    // Light volumes + floor patches + labels
    for (let i = 0; i < NUM_WINDOWS; i++) {{
        const wv = d.window_volumes[i];
        const volIdx = volStartIdx + i;
        const flIdx = floorStartIdx + i;
        const lblIdx = labelStartIdx + i;

        if (wv.is_facing && wv.volume_verts) {{
            const top = wv.volume_verts.top;
            const bot = wv.volume_verts.bottom;
            const vx = top.map(p => p[0]).concat(bot.map(p => p[0]));
            const vy = top.map(p => p[1]).concat(bot.map(p => p[1]));
            const vz = top.map(p => p[2]).concat(bot.map(p => p[2]));
            const vColor = wv.plant_hit ? '#ff9800' : '#ffeb3b';
            const vOpacity = wv.plant_hit ? 0.2 : 0.1;
            // Frustum triangles: 4 sides * 2 + top * 2 + bottom * 2 = 12
            const fi = [0,0, 4,4, 0,0, 1,1, 2,2, 3,3];
            const fj = [1,2, 5,6, 1,5, 2,6, 3,7, 0,4];
            const fk = [3,3, 4,4, 5,4, 6,5, 7,6, 4,7];
            restyles.push([{{
                'x': [vx], 'y': [vy], 'z': [vz],
                'i': [fi], 'j': [fj], 'k': [fk],
                'color': [vColor], 'opacity': [vOpacity], 'visible': [true]
            }}, [volIdx]]);

            // Floor patch
            const fp = wv.floor_patch;
            const fpx = fp.map(p => p[0]);
            const fpy = fp.map(p => p[1]);
            const fpz = fp.map(p => p[2]);
            const fpColor = wv.plant_hit ? '#ffab00' : '#fff9c4';
            restyles.push([{{
                'x': [fpx], 'y': [fpy], 'z': [fpz],
                'color': [fpColor], 'visible': [true]
            }}, [flIdx]]);
        }} else {{
            restyles.push([{{ 'visible': [false] }}, [volIdx]]);
            restyles.push([{{ 'visible': [false] }}, [flIdx]]);
        }}

        // Window label
        // Position label at window center + small offset
        const wObj = wv;
        // We need window center positions — derive from volume top corners or use a fallback
        let lx = 0, ly = 0, lz = 0;
        if (wv.volume_verts) {{
            const top = wv.volume_verts.top;
            lx = (top[0][0] + top[1][0] + top[2][0] + top[3][0]) / 4;
            ly = (top[0][1] + top[1][1] + top[2][1] + top[3][1]) / 4;
            lz = (top[0][2] + top[1][2] + top[2][2] + top[3][2]) / 4 + 0.3;
        }}
        const labelText = wv.plant_hit ? '☀️HIT' : (wv.is_facing ? '🌤️' : '');
        restyles.push([{{
            'x': [[lx]], 'y': [[ly]], 'z': [[lz]],
            'text': [[labelText]]
        }}, [lblIdx]]);

        // Update grid cell
        const cell = document.getElementById('wc-' + wv.window_id);
        if (cell) {{
            if (wv.plant_hit) {{
                cell.className = 'win-cell win-hit';
                cell.textContent = wv.window_id.replace('window_', 'W') + ' ☀️HIT';
            }} else if (wv.is_facing) {{
                cell.className = 'win-cell win-sun';
                cell.textContent = wv.window_id.replace('window_', 'W') + ' 🌤️';
            }} else {{
                cell.className = 'win-cell win-none';
                cell.textContent = wv.window_id.replace('window_', 'W');
            }}
        }}
    }}

    // Hit rays
    const rays = d.hit_rays || [];
    for (let r = 0; r < MAX_RAYS; r++) {{
        const rIdx = rayStartIdx + r;
        if (r < rays.length) {{
            const ray = rays[r];
            restyles.push([{{
                'x': [[ray.start[0], ray.end[0]]],
                'y': [[ray.start[1], ray.end[1]]],
                'z': [[ray.start[2], ray.end[2]]],
                'visible': [true]
            }}, [rIdx]]);
        }} else {{
            restyles.push([{{ 'visible': [false] }}, [rIdx]]);
        }}
    }}

    // Apply all restyles
    const graphDiv = document.getElementById('plot');
    for (const [upd, traceIdx] of restyles) {{
        Plotly.restyle(graphDiv, upd, traceIdx);
    }}
}}

slider.addEventListener('input', () => updateTime(parseInt(slider.value)));

// Play/pause
let playing = false;
let playInterval = null;
playBtn.addEventListener('click', () => {{
    playing = !playing;
    playBtn.textContent = playing ? '⏸' : '▶';
    if (playing) {{
        playInterval = setInterval(() => {{
            let v = parseInt(slider.value) + 1;
            if (v >= timeData.length) v = 0;
            slider.value = v;
            updateTime(v);
        }}, 200);
    }} else {{
        clearInterval(playInterval);
    }}
}});

// Initial render
if (timeData.length > 0) {{
    // Find a good starting index (first hit or noon)
    let startIdx = 0;
    for (let i = 0; i < timeData.length; i++) {{
        if (timeData[i].any_hit) {{ startIdx = i; break; }}
    }}
    if (startIdx === 0) startIdx = Math.floor(timeData.length / 2);
    slider.value = startIdx;
    updateTime(startIdx);
}}
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Parse date argument
    if len(sys.argv) > 1:
        target_date = date.fromisoformat(sys.argv[1])
    else:
        target_date = date.today()

    config_path = ROOT / "config" / "default_config.json"
    config, location, wall_lengths = load_config_and_location(str(config_path))

    print(f"Generating visualization for {target_date} ...")
    print(f"  Location: ({location['latitude']}, {location['longitude']}) "
          f"TZ={location.get('timezone_name', 'N/A')}")
    print(f"  Plant at ({config.plant.center_x:.2f}, {config.plant.center_y:.2f})")
    print(f"  Windows: {len(config.windows)}")

    # Generate sun data at 5-minute intervals
    sun_data = generate_sun_data_for_date(
        latitude=location["latitude"],
        longitude=location["longitude"],
        target_date=target_date,
        timezone_name=location.get("timezone_name"),
        timezone_offset=location.get("timezone_offset", -5),
        interval_minutes=5,
    )
    print(f"  Sun data points: {len(sun_data)}")

    # Pre-compute all time-varying geometry
    time_data, sample_pts = precompute_time_data(config, sun_data, wall_lengths)
    hit_count = sum(1 for t in time_data if t["any_hit"])
    print(f"  Timesteps with plant hit: {hit_count}/{len(time_data)}")

    # Build static traces
    window_ids = [w.id for w in config.windows]
    static_traces = []
    static_traces.append(build_floor_trace(wall_lengths))
    static_traces.extend(build_wall_traces(config, wall_lengths))
    static_traces.extend(build_window_traces(config))
    static_traces.append(build_plant_trace(config.plant))
    static_traces.append(build_sample_points_trace(sample_pts))

    # Generate HTML
    html = generate_html(
        static_traces, time_data, sample_pts, window_ids,
        str(target_date), wall_lengths, config.plant,
    )

    # Write output
    output_dir = ROOT / "examples"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "room_visualization.html"
    output_path.write_text(html, encoding="utf-8")

    size_kb = output_path.stat().st_size / 1024
    print(f"\n  Output: {output_path} ({size_kb:.0f} KB)")
    print("  Open in a browser to interact with the 3D visualization.")


if __name__ == "__main__":
    main()
