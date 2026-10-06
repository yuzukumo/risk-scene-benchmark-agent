"""
Render academic-quality BEV scenario visualization.

Replaces the old visualization with publication-ready style.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, FancyBboxPatch, Circle
from matplotlib.collections import LineCollection

from nusc_scene_agent.geometry import global_xy_to_anchor_ego, oriented_box_corners
from nusc_scene_agent.models import ValidatedCase

# Import academic style
import sys
sys.path.insert(0, str(Path(__file__).parent))
from academic_figure_style import set_academic_style, COLORS

set_academic_style(context="paper")

# Academic color scheme for BEV
BEV_COLORS = {
    "background": "#FAFAFA",
    "grid": "#E8E8E8",
    "axis": "#9E9E9E",
    "ego": COLORS["gray"],
    "ego_trajectory": "#2C2C2C",
    "target": COLORS["red"],
    "target_trajectory": "#D32F2F",
    "context": "#BDBDBD",
    "drivable": "#E3F2FD",
    "lane": "#90CAF9",
    "connector": "#64B5F6",
    "crosswalk": COLORS["green"],
    "walkway": "#C8E6C9",
    "stop_line": COLORS["red"],
}


def _draw_box(
    ax: plt.Axes,
    x: float, y: float,
    width: float, length: float,
    yaw: float,
    color: str,
    alpha: float,
    lw: float,
    fill: bool = True,
) -> None:
    """Draw oriented bounding box with academic styling."""
    corners = oriented_box_corners(x, y, width, length, yaw)

    if fill:
        patch = Polygon(
            corners, closed=True,
            facecolor=color, edgecolor="#2C2C2C",
            linewidth=lw, alpha=alpha
        )
    else:
        patch = Polygon(
            corners, closed=True,
            facecolor="none", edgecolor=color,
            linewidth=lw, alpha=alpha
        )
    ax.add_patch(patch)


def _draw_map_layers(ax: plt.Axes, case: ValidatedCase) -> None:
    """Draw HD map layers with academic color scheme."""
    if not case.map_geometries:
        return

    # Drivable area (lightest)
    for polygon in case.map_geometries.get("drivable_area", []):
        ax.fill(
            polygon[:, 0], polygon[:, 1],
            facecolor=BEV_COLORS["drivable"],
            edgecolor="none",
            alpha=0.4,
            zorder=1
        )

    # Lanes
    for polygon in case.map_geometries.get("lane", []):
        patch = Polygon(
            polygon, closed=True,
            facecolor=BEV_COLORS["lane"],
            edgecolor=BEV_COLORS["lane"],
            linewidth=0.6,
            alpha=0.3
        )
        patch.set_zorder(2)
        ax.add_patch(patch)

    # Lane connectors
    for polygon in case.map_geometries.get("lane_connector", []):
        patch = Polygon(
            polygon, closed=True,
            facecolor=BEV_COLORS["connector"],
            edgecolor=BEV_COLORS["connector"],
            linewidth=0.6,
            alpha=0.25
        )
        patch.set_zorder(2.1)
        ax.add_patch(patch)

    # Walkways
    for polygon in case.map_geometries.get("walkway", []):
        patch = Polygon(
            polygon, closed=True,
            facecolor=BEV_COLORS["walkway"],
            edgecolor="none",
            alpha=0.4
        )
        patch.set_zorder(2.2)
        ax.add_patch(patch)

    # Crosswalks (prominent)
    for polygon in case.map_geometries.get("ped_crossing", []):
        patch = Polygon(
            polygon, closed=True,
            facecolor=BEV_COLORS["crosswalk"],
            edgecolor=BEV_COLORS["crosswalk"],
            linewidth=0.8,
            alpha=0.5
        )
        patch.set_zorder(2.3)
        ax.add_patch(patch)

    # Stop lines
    for polygon in case.map_geometries.get("stop_line", []):
        ax.plot(
            polygon[:, 0], polygon[:, 1],
            color=BEV_COLORS["stop_line"],
            linewidth=1.5,
            alpha=0.8,
            zorder=2.4
        )


def _draw_scale_bar(ax: plt.Axes, x: float, y: float, length: float = 10.0) -> None:
    """Draw metric scale bar."""
    ax.plot(
        [x, x + length], [y, y],
        color="#2C2C2C", linewidth=2, solid_capstyle="butt"
    )
    # End caps
    ax.plot([x, x], [y - 0.3, y + 0.3], color="#2C2C2C", linewidth=2)
    ax.plot([x + length, x + length], [y - 0.3, y + 0.3], color="#2C2C2C", linewidth=2)

    # Label
    ax.text(
        x + length / 2, y - 1.2,
        f"{int(length)} m",
        ha="center", va="top",
        fontsize=8,
        color="#2C2C2C"
    )


def _draw_compass(ax: plt.Axes, x: float, y: float, radius: float = 2.0) -> None:
    """Draw north arrow compass."""
    # Arrow
    ax.annotate(
        "", xy=(x, y + radius), xytext=(x, y),
        arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6",
                       color="#2C2C2C", linewidth=1.5)
    )
    # Label
    ax.text(
        x, y + radius + 0.8,
        "N",
        ha="center", va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#2C2C2C"
    )


def render_bev_evidence(case: ValidatedCase, output_path: Path) -> None:
    """
    Render publication-quality BEV visualization.

    Args:
        case: Validated scenario case
        output_path: Output PNG path
    """

    # Create figure with academic dimensions
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=100)

    # Background and grid
    ax.set_facecolor(BEV_COLORS["background"])
    ax.grid(True, color=BEV_COLORS["grid"], linewidth=0.5, alpha=0.8, zorder=0)

    # Origin axes (ego reference frame)
    ax.axhline(0.0, color=BEV_COLORS["axis"], linewidth=1.0, alpha=0.6, zorder=0.5)
    ax.axvline(0.0, color=BEV_COLORS["axis"], linewidth=1.0, alpha=0.6, zorder=0.5)

    # Draw map layers
    _draw_map_layers(ax, case)

    # Get anchor frame
    anchor_row = case.timeline.loc[case.timeline["sample_token"] == case.candidate.sample_token].iloc[0]
    anchor_xy = np.asarray([anchor_row["ego_x"], anchor_row["ego_y"]], dtype=float)
    anchor_yaw = float(anchor_row["ego_yaw"])

    # Ego trajectory
    ego_points_global = case.ego_window[["ego_x", "ego_y"]].to_numpy(dtype=float)
    ego_points = global_xy_to_anchor_ego(ego_points_global, anchor_xy, anchor_yaw)

    ax.plot(
        ego_points[:, 0], ego_points[:, 1],
        color=BEV_COLORS["ego_trajectory"],
        linewidth=2.5,
        label="Ego",
        zorder=10,
        solid_capstyle="round"
    )

    # Context agents (limit to 24 for clarity)
    for _, row in case.context_agents.head(24).iterrows():
        if row["instance_token"] == case.candidate.instance_token:
            continue

        rel_x, rel_y = global_xy_to_anchor_ego(
            np.asarray([[row["x"], row["y"]]], dtype=float),
            anchor_xy, anchor_yaw
        )[0]
        rel_yaw = float(row["yaw"]) - anchor_yaw

        _draw_box(
            ax, rel_x, rel_y,
            float(row["width"]), float(row["length"]),
            rel_yaw,
            BEV_COLORS["context"],
            alpha=0.6,
            lw=0.8,
            fill=True
        )

    # Target agent trajectory
    target_mask = case.timeline["instance_token"] == case.candidate.instance_token
    target_points_global = case.timeline.loc[target_mask, ["x", "y"]].to_numpy(dtype=float)
    target_points = global_xy_to_anchor_ego(target_points_global, anchor_xy, anchor_yaw)

    if len(target_points) > 1:
        ax.plot(
            target_points[:, 0], target_points[:, 1],
            color=BEV_COLORS["target_trajectory"],
            linewidth=2.0,
            linestyle="--",
            label="Target",
            zorder=11,
            alpha=0.9
        )

    # Target agent box at anchor frame
    target_anchor = case.timeline.loc[
        (case.timeline["instance_token"] == case.candidate.instance_token) &
        (case.timeline["sample_token"] == case.candidate.sample_token)
    ].iloc[0]

    target_rel = global_xy_to_anchor_ego(
        np.asarray([[target_anchor["x"], target_anchor["y"]]], dtype=float),
        anchor_xy, anchor_yaw
    )[0]
    target_rel_yaw = float(target_anchor["yaw"]) - anchor_yaw

    _draw_box(
        ax,
        target_rel[0], target_rel[1],
        float(target_anchor["width"]), float(target_anchor["length"]),
        target_rel_yaw,
        BEV_COLORS["target"],
        alpha=0.9,
        lw=1.2,
        fill=True
    )

    # Ego vehicle box at origin
    _draw_box(
        ax, 0, 0,
        case.candidate.ego_width, case.candidate.ego_length,
        0.0,
        BEV_COLORS["ego"],
        alpha=0.9,
        lw=1.2,
        fill=True
    )

    # Set viewing window
    view_range = 40  # meters
    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_aspect("equal")

    # Labels
    ax.set_xlabel("Lateral offset (m)", fontsize=9)
    ax.set_ylabel("Longitudinal offset (m)", fontsize=9)

    # Legend
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#E0E0E0",
        framealpha=0.95,
        fontsize=8
    )

    # Scale bar and compass
    _draw_scale_bar(ax, -35, -35)
    _draw_compass(ax, 35, -35)

    # Title
    behavior = case.candidate.behavior_label.replace("_", " ").title()
    ax.set_title(
        f"Scenario: {behavior}",
        fontsize=10,
        fontweight="bold",
        pad=10
    )

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
