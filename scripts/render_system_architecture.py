"""
Render academic-quality system architecture diagram.

Clean, publication-ready visualization of the pipeline flow.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from academic_figure_style import set_academic_style, COLORS, save_figure

set_academic_style(context="paper")


def draw_module(ax, x, y, width, height, title, subtitle, color, text_color="#2C2C2C"):
    """Draw a rounded module box with title and subtitle."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor=text_color,
        linewidth=1.2,
        alpha=0.9,
    )
    ax.add_patch(box)

    # Title
    ax.text(
        x + width / 2, y + height * 0.65,
        title,
        ha="center", va="center",
        fontsize=9, fontweight="bold",
        color=text_color,
    )

    # Subtitle
    ax.text(
        x + width / 2, y + height * 0.35,
        subtitle,
        ha="center", va="center",
        fontsize=7.5,
        color=text_color,
        alpha=0.85,
    )


def draw_arrow(ax, x1, y1, x2, y2, label="", color="#5D5D5D"):
    """Draw curved arrow with optional label."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_width=0.15,head_length=0.2",
        color=color,
        linewidth=1.5,
        connectionstyle="arc3,rad=0.0",
        alpha=0.8,
    )
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mid_x, mid_y + 0.08,
            label,
            ha="center", va="bottom",
            fontsize=7,
            color=color,
            style="italic",
        )


def main():
    fig = plt.figure(figsize=(7.2, 4.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Title
    ax.text(
        5.0, 5.7,
        "Risk-Scenario Mining and End-to-End Planner Evaluation",
        ha="center", va="top",
        fontsize=11, fontweight="bold",
    )

    # Column headers
    header_y = 5.3
    ax.text(1.4, header_y, "Data Sources", ha="center", fontsize=8.5, color=COLORS["gray"], style="italic")
    ax.text(5.0, header_y, "Methods", ha="center", fontsize=8.5, color=COLORS["gray"], style="italic")
    ax.text(8.6, header_y, "Evaluation", ha="center", fontsize=8.5, color=COLORS["gray"], style="italic")

    # Module dimensions
    data_w, method_w, eval_w = 1.8, 2.6, 1.8
    box_h = 0.65

    # Row 1: Scenario Mining (Green)
    y1 = 4.3
    draw_module(ax, 0.5, y1, data_w, box_h, "nuScenes", "Sensor logs + HD maps", "#E8F5E9", COLORS["green"])
    draw_module(ax, 3.7, y1, method_w, box_h, "Scenario Mining", "NL query + validation", "#C8E6C9", COLORS["green"])
    draw_module(ax, 7.7, y1, eval_w, box_h, "Risk Slices", "Forecast + occupancy", "#A5D6A7", COLORS["green"])

    draw_arrow(ax, 2.3, y1 + box_h / 2, 3.7, y1 + box_h / 2)
    draw_arrow(ax, 6.3, y1 + box_h / 2, 7.7, y1 + box_h / 2)

    # Row 2: Planner Training (Blue)
    y2 = 3.3
    draw_module(ax, 0.5, y2, data_w, box_h, "Bench2Drive", "Train / val / test split", "#E3F2FD", COLORS["blue"])
    draw_module(ax, 3.7, y2, method_w, box_h, "E2E Training", "Ablations × 3 seeds", "#BBDEFB", COLORS["blue"])
    draw_module(ax, 7.7, y2, eval_w, box_h, "Logged Replay", "Open-loop metrics", "#90CAF9", COLORS["blue"])

    draw_arrow(ax, 2.3, y2 + box_h / 2, 3.7, y2 + box_h / 2)
    draw_arrow(ax, 6.3, y2 + box_h / 2, 7.7, y2 + box_h / 2)

    # Data selection feedback arrow
    draw_arrow(ax, 5.0, y1, 5.0, y2 + box_h, "data selection", COLORS["orange"])

    # Row 3: CARLA Evaluation (Orange)
    y3 = 2.3
    draw_module(ax, 0.5, y3, data_w, box_h, "CARLA", "Live cameras + traffic", "#FFF3E0", COLORS["orange"])
    draw_module(ax, 3.7, y3, method_w, box_h, "Closed-Loop Sim", "Fixed routes × 3 seeds", "#FFE0B2", COLORS["orange"])
    draw_module(ax, 7.7, y3, eval_w, box_h, "Safety Metrics", "Collisions + completion", "#FFCC80", COLORS["orange"])

    draw_arrow(ax, 2.3, y3 + box_h / 2, 3.7, y3 + box_h / 2)
    draw_arrow(ax, 6.3, y3 + box_h / 2, 7.7, y3 + box_h / 2)

    # Checkpoint flow arrow
    draw_arrow(ax, 5.0, y2, 5.0, y3 + box_h, "checkpoint", COLORS["gray"])

    # Row 4: nuPlan Diagnostics (Brown)
    y4 = 1.3
    draw_module(ax, 0.5, y4, data_w, box_h, "nuPlan", "Ego + actor logs", "#EFEBE9", COLORS["brown"])
    draw_module(ax, 3.7, y4, method_w, box_h, "Replay Baselines", "Kinematic profiles", "#D7CCC8", COLORS["brown"])
    draw_module(ax, 7.7, y4, eval_w, box_h, "Failure Analysis", "Paired bootstrap CI", "#BCAAA4", COLORS["brown"])

    draw_arrow(ax, 2.3, y4 + box_h / 2, 3.7, y4 + box_h / 2)
    draw_arrow(ax, 6.3, y4 + box_h / 2, 7.7, y4 + box_h / 2)

    # Legend / Key findings box
    legend_y = 0.3
    ax.text(
        5.0, legend_y,
        "Reproducible protocol: Archive-disjoint splits · Paired statistical tests · SHA256 provenance",
        ha="center", va="center",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#BDBDBD", linewidth=0.8),
        color=COLORS["gray"],
    )

    plt.tight_layout(pad=0.2)
    save_figure(fig, "assets/pipeline_overview.png", dpi=600)
    print("✓ Generated: assets/pipeline_overview.png")


if __name__ == "__main__":
    main()
