"""Plotly 3D visualization for the sun-plant simulation.

This module provides functions to create interactive 3D visualizations
of the room geometry, windows, plant, and sun rays.
"""

import math
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from ..core.geometry import sun_direction_from_angles
from ..core.models import Config, HitResult, Plant, Window


def create_window_mesh(window: Window, color: str = "lightblue", opacity: float = 0.5) -> go.Mesh3d:
    """Create a Plotly mesh for a window rectangle.

    Args:
        window: The window to visualize.
        color: Fill color for the window.
        opacity: Transparency (0-1).

    Returns:
        Plotly Mesh3d trace.
    """
    corners = window.get_corners()

    # Extract coordinates
    x = [c[0] for c in corners]
    y = [c[1] for c in corners]
    z = [c[2] for c in corners]

    # Define two triangles for the rectangle
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color=color,
        opacity=opacity,
        name=f"Window {window.id}",
        showlegend=True,
    )


def create_window_frame(window: Window, color: str = "blue", width: int = 3) -> go.Scatter3d:
    """Create a Plotly line trace for the window frame.

    Args:
        window: The window to outline.
        color: Line color.
        width: Line width.

    Returns:
        Plotly Scatter3d trace.
    """
    corners = window.get_corners()
    # Close the rectangle by repeating the first corner
    corners.append(corners[0])

    x = [c[0] for c in corners]
    y = [c[1] for c in corners]
    z = [c[2] for c in corners]

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color=color, width=width),
        name=f"Frame {window.id}",
        showlegend=False,
    )


def create_plant_cylinder(
    plant: Plant,
    color: str = "green",
    opacity: float = 0.7,
    n_points: int = 20,
) -> go.Mesh3d:
    """Create a Plotly mesh for the plant cylinder.

    Args:
        plant: The plant to visualize.
        color: Fill color.
        opacity: Transparency.
        n_points: Number of points around the circumference.

    Returns:
        Plotly Mesh3d trace.
    """
    # Generate points around the top and bottom circles
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Bottom circle
    x_bottom = plant.center_x + plant.radius * np.cos(theta)
    y_bottom = plant.center_y + plant.radius * np.sin(theta)
    z_bottom = np.full(n_points, plant.z_min)

    # Top circle
    x_top = plant.center_x + plant.radius * np.cos(theta)
    y_top = plant.center_y + plant.radius * np.sin(theta)
    z_top = np.full(n_points, plant.z_max)

    # Combine all vertices
    x = np.concatenate([x_bottom, x_top])
    y = np.concatenate([y_bottom, y_top])
    z = np.concatenate([z_bottom, z_top])

    # Create triangles for the cylinder surface
    i_list = []
    j_list = []
    k_list = []

    for idx in range(n_points):
        next_idx = (idx + 1) % n_points

        # Two triangles per side segment
        # Triangle 1: bottom[idx], bottom[next], top[idx]
        i_list.append(idx)
        j_list.append(next_idx)
        k_list.append(idx + n_points)

        # Triangle 2: bottom[next], top[next], top[idx]
        i_list.append(next_idx)
        j_list.append(next_idx + n_points)
        k_list.append(idx + n_points)

    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i_list,
        j=j_list,
        k=k_list,
        color=color,
        opacity=opacity,
        name="Plant",
        showlegend=True,
    )


def create_sun_indicator(
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    center: tuple[float, float, float] = (3.0, 2.5, 3.0),
    distance: float = 5.0,
    size: int = 15,
) -> go.Scatter3d:
    """Create a marker showing the sun's position/direction.

    Args:
        sun_azimuth_deg: Sun azimuth in degrees.
        sun_elevation_deg: Sun elevation in degrees.
        center: Reference point for the sun indicator.
        distance: Distance from center to place the indicator.
        size: Marker size.

    Returns:
        Plotly Scatter3d trace.
    """
    sun_dir = sun_direction_from_angles(sun_azimuth_deg, sun_elevation_deg)
    sun_pos = np.array(center) + distance * sun_dir

    return go.Scatter3d(
        x=[sun_pos[0]],
        y=[sun_pos[1]],
        z=[sun_pos[2]],
        mode="markers",
        marker=dict(size=size, color="yellow", symbol="circle"),
        name=f"Sun (az={sun_azimuth_deg:.0f}°, el={sun_elevation_deg:.0f}°)",
        showlegend=True,
    )


def create_sun_rays(
    plant: Plant,
    hit_result: HitResult,
    ray_length: float = 6.0,
    hit_color: str = "gold",
    miss_color: str = "gray",
) -> list[go.Scatter3d]:
    """Create line traces showing sun rays from plant to windows.

    Args:
        plant: The plant geometry.
        hit_result: Result from hit test (contains hit points and sun direction).
        ray_length: Length of ray lines to draw.
        hit_color: Color for rays that hit through windows.
        miss_color: Color for rays that don't hit.

    Returns:
        List of Plotly Scatter3d traces.
    """
    traces = []

    if hit_result.sun_direction is None:
        return traces

    sun_dir = hit_result.sun_direction
    hit_points_set = set(tuple(p.tolist()) for p in hit_result.hit_points)

    # Draw rays from hit points
    for point in hit_result.hit_points:
        end = point + ray_length * sun_dir
        traces.append(
            go.Scatter3d(
                x=[point[0], end[0]],
                y=[point[1], end[1]],
                z=[point[2], end[2]],
                mode="lines",
                line=dict(color=hit_color, width=4),
                name="Hit ray",
                showlegend=False,
            )
        )

    return traces


def create_sample_points_markers(
    plant: Plant,
    hit_result: Optional[HitResult] = None,
    n_angular: int = 8,
    n_vertical: int = 3,
) -> go.Scatter3d:
    """Create markers showing the sample points on the plant.

    Args:
        plant: The plant geometry.
        hit_result: Optional hit result to color points by hit status.
        n_angular: Angular sampling resolution.
        n_vertical: Vertical sampling resolution.

    Returns:
        Plotly Scatter3d trace.
    """
    from ..core.hit_test import generate_plant_sample_points

    points = generate_plant_sample_points(plant, n_angular, n_vertical)

    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]

    # Color based on hit status if available
    if hit_result and hit_result.hit_points:
        hit_set = set(tuple(p.tolist()) for p in hit_result.hit_points)
        colors = ["gold" if tuple(p.tolist()) in hit_set else "darkgreen" for p in points]
    else:
        colors = ["darkgreen"] * len(points)

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(size=6, color=colors),
        name="Sample points",
        showlegend=True,
    )


def create_coordinate_axes(
    origin: tuple[float, float, float] = (0, 0, 0),
    length: float = 1.0,
) -> list[go.Scatter3d]:
    """Create coordinate axis indicators.

    Args:
        origin: Origin point for axes.
        length: Length of each axis line.

    Returns:
        List of Plotly Scatter3d traces (X=red, Y=green, Z=blue).
    """
    ox, oy, oz = origin
    return [
        go.Scatter3d(
            x=[ox, ox + length],
            y=[oy, oy],
            z=[oz, oz],
            mode="lines+text",
            line=dict(color="red", width=4),
            text=["", "E"],
            textposition="top center",
            name="East (X)",
            showlegend=False,
        ),
        go.Scatter3d(
            x=[ox, ox],
            y=[oy, oy + length],
            z=[oz, oz],
            mode="lines+text",
            line=dict(color="green", width=4),
            text=["", "N"],
            textposition="top center",
            name="North (Y)",
            showlegend=False,
        ),
        go.Scatter3d(
            x=[ox, ox],
            y=[oy, oy],
            z=[oz, oz + length],
            mode="lines+text",
            line=dict(color="blue", width=4),
            text=["", "Up"],
            textposition="top center",
            name="Up (Z)",
            showlegend=False,
        ),
    ]


def build_scene(
    config: Config,
    sun_azimuth_deg: Optional[float] = None,
    sun_elevation_deg: Optional[float] = None,
    hit_result: Optional[HitResult] = None,
    show_sample_points: bool = True,
    show_rays: bool = True,
    show_axes: bool = True,
    title: str = "Sun-Plant Hit Visualization",
) -> go.Figure:
    """Build a complete Plotly 3D scene with room geometry.

    Args:
        config: Configuration with windows and plant geometry.
        sun_azimuth_deg: Sun azimuth for indicator and rays.
        sun_elevation_deg: Sun elevation for indicator and rays.
        hit_result: Optional hit result to show rays and highlight hits.
        show_sample_points: Whether to show plant sample points.
        show_rays: Whether to show sun rays (requires hit_result).
        show_axes: Whether to show coordinate axes.
        title: Plot title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # Add coordinate axes
    if show_axes:
        for trace in create_coordinate_axes():
            fig.add_trace(trace)

    # Add windows
    for window in config.windows:
        fig.add_trace(create_window_mesh(window))
        fig.add_trace(create_window_frame(window))

    # Add plant cylinder
    fig.add_trace(create_plant_cylinder(config.plant))

    # Add sample points
    if show_sample_points:
        fig.add_trace(
            create_sample_points_markers(
                config.plant,
                hit_result,
                config.simulation.sample_points_angular,
                config.simulation.sample_points_vertical,
            )
        )

    # Add sun indicator
    if sun_azimuth_deg is not None and sun_elevation_deg is not None:
        center = (config.plant.center_x, config.plant.center_y, config.plant.z_max)
        fig.add_trace(create_sun_indicator(sun_azimuth_deg, sun_elevation_deg, center))

    # Add sun rays
    if show_rays and hit_result is not None:
        for trace in create_sun_rays(config.plant, hit_result):
            fig.add_trace(trace)

    # Configure layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Up (m)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
            ),
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def visualize_hit_test(
    config: Config,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
) -> go.Figure:
    """Convenience function to visualize a single hit test.

    Runs the hit test and creates a visualization with the results.

    Args:
        config: Room configuration.
        sun_azimuth_deg: Sun azimuth.
        sun_elevation_deg: Sun elevation.

    Returns:
        Plotly Figure with hit test visualization.
    """
    from ..core.hit_test import check_sun_hits_plant

    hit_result = check_sun_hits_plant(
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        plant=config.plant,
        windows=config.windows,
        n_angular=config.simulation.sample_points_angular,
        n_vertical=config.simulation.sample_points_vertical,
    )

    status = "HIT" if hit_result.is_hit else "MISS"
    title = f"Sun-Plant Test: {status} (az={sun_azimuth_deg:.0f}°, el={sun_elevation_deg:.0f}°)"

    return build_scene(
        config=config,
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        hit_result=hit_result,
        title=title,
    )
