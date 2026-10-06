from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 14), ylim=(0, 6.2))
    ax.axis("off")
    ax.text(0.35, 5.85, "Risk scenarios, data selection and planner evaluation", fontsize=18, weight="bold")
    ax.text(0.35, 5.48, "Data sources", fontsize=10, color="#555555")
    ax.text(4.1, 5.48, "Models and protocols", fontsize=10, color="#555555")
    ax.text(9.6, 5.48, "Evidence", fontsize=10, color="#555555")

    def node(x, y, width, title, body, color):
        ax.add_patch(Rectangle((x, y), width, 0.84, facecolor="#f8f9fa", edgecolor="#cbd0d4", linewidth=0.8))
        ax.add_patch(Rectangle((x, y), 0.05, 0.84, facecolor=color, edgecolor="none"))
        ax.text(x + 0.16, y + 0.55, title, fontsize=11, weight="bold", va="center")
        ax.text(x + 0.16, y + 0.24, body, fontsize=9, va="center", color="#42464a")

    def arrow(start, end, label=""):
        ax.annotate("", end, start, arrowprops={"arrowstyle": "->", "color": "#777d81", "lw": 1.1})
        if label:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, label,
                    fontsize=8, color="#555555", ha="center", va="bottom")

    node(0.35, 4.2, 2.9, "nuScenes", "Logs, actors and map context", "#237f68")
    node(4.1, 4.2, 4.4, "Scenario mining", "Local query planning + deterministic validation", "#237f68")
    node(9.55, 4.2, 4.1, "Risk slices", "Grounded anchors; forecast + occupancy adapters", "#237f68")
    arrow((3.25, 4.62), (4.1, 4.62))
    arrow((8.5, 4.62), (9.55, 4.62))
    node(0.35, 2.97, 2.9, "Bench2Drive logs", "Archive-disjoint train / val / test", "#326da0")
    node(4.1, 2.97, 4.4, "Controlled E2E training", "Matched budgets, input/loss ablations, 3 seeds", "#326da0")
    node(9.55, 2.97, 4.1, "Held-out planner diagnostics", "Open-loop + fixed logged-sensor replay", "#326da0")
    arrow((3.25, 3.39), (4.1, 3.39))
    arrow((8.5, 3.39), (9.55, 3.39))
    arrow((6.3, 4.2), (6.3, 3.81))
    ax.text(6.48, 4.0, "taxonomy prior vs. random sampling", fontsize=8, va="center", color="#555555")
    node(0.35, 1.74, 2.9, "CARLA", "Live cameras + natural vehicle traffic", "#a14d59")
    node(4.1, 1.74, 4.4, "Fixed simulator evaluation", "Predefined routes and seeds; all attempts retained", "#a14d59")
    node(9.55, 1.74, 4.1, "Driving outcomes", "Completion, collisions and control attribution", "#a14d59")
    arrow((3.25, 2.16), (4.1, 2.16))
    arrow((8.5, 2.16), (9.55, 2.16))
    arrow((6.3, 2.97), (6.3, 2.58))
    ax.text(6.48, 2.77, "checkpoint", fontsize=8, va="center", color="#555555")
    node(0.35, 0.51, 2.9, "nuPlan", "Logged ego, actors and traffic lights", "#8c7136")
    node(4.1, 0.51, 4.4, "Replay baselines", "Kinematics and following-controller diagnostics", "#8c7136")
    node(9.55, 0.51, 4.1, "Failure analysis and provenance", "Scenario breakdowns, paired statistics, SHA256", "#8c7136")
    arrow((3.25, 0.93), (4.1, 0.93))
    arrow((8.5, 0.93), (9.55, 0.93))
    fig.tight_layout(pad=0.4)
    fig.savefig(Path("assets/pipeline_overview.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
