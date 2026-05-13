"""Generate the four Medium-article images.

Run once. Output lands in `medium/images/`.

* training_curves.png — from real `data/models/gnn/history.json`
* pr_curves.png       — copy of `data/models/ensemble/eval/test_pr.png` (real eval)
* architecture.png    — system architecture diagram, drawn with matplotlib
* graph_schema.png    — heterogeneous graph schema diagram, drawn with matplotlib
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

# Paths
HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[3]  # walk up from medium/ in worktree to fraud-detection-gnn
DATA = PROJ / "data"
OUT = HERE / "images"
OUT.mkdir(parents=True, exist_ok=True)

# Shared palette (slate / red / cyan / amber, matches the dashboard mood)
PALETTE = {
    "bg": "#0f172a",
    "panel": "#1e293b",
    "panel_alt": "#334155",
    "border": "#475569",
    "text": "#f1f5f9",
    "muted": "#cbd5e1",
    "red": "#dc2626",
    "red_light": "#f87171",
    "blue": "#3b82f6",
    "blue_dark": "#1d4ed8",
    "cyan": "#06b6d4",
    "amber": "#f59e0b",
    "green": "#22c55e",
    "purple": "#a855f7",
}

# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def make_training_curves() -> None:
    hist_path = DATA / "models" / "gnn" / "history.json"
    hist = json.load(hist_path.open())
    epochs = np.array([h["epoch"] for h in hist])
    train_loss = np.array([h["train_loss"] for h in hist])
    val_auprc = np.array([h["val_auprc"] for h in hist])
    val_auroc = np.array([h["val_auroc"] for h in hist])

    best_epoch = int(np.argmax(val_auprc))
    best_auprc = float(val_auprc[best_epoch])

    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=160, facecolor="white")
    ax1.set_facecolor("#fafafa")
    ax1.grid(True, alpha=0.3, linestyle="--")

    line_auprc, = ax1.plot(
        epochs, val_auprc,
        color=PALETTE["red"], linewidth=2.4, label="Validation AUPRC",
    )
    line_auroc, = ax1.plot(
        epochs, val_auroc,
        color=PALETTE["purple"], linewidth=1.8, alpha=0.75,
        linestyle="--", label="Validation AUROC",
    )
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Validation metric", fontsize=11)
    ax1.set_ylim(0, 1.0)

    ax2 = ax1.twinx()
    line_loss, = ax2.plot(
        epochs, train_loss,
        color=PALETTE["blue_dark"], linewidth=1.8, alpha=0.7,
        label="Train loss",
    )
    ax2.set_ylabel("Train loss", fontsize=11, color=PALETTE["blue_dark"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["blue_dark"])

    ax1.axvline(best_epoch, color=PALETTE["green"], linestyle=":", alpha=0.9, linewidth=1.6)
    ax1.annotate(
        f"best epoch: {best_epoch}\nval AUPRC = {best_auprc:.3f}",
        xy=(best_epoch, best_auprc),
        xytext=(best_epoch + 8, best_auprc + 0.08),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=PALETTE["green"], linewidth=1.2),
        arrowprops=dict(arrowstyle="->", color=PALETTE["green"], lw=1.2),
    )

    plt.title(
        f"Meshwatch GNN training curves (n={len(epochs)} epochs, full 590k run)",
        fontsize=13, fontweight="bold", pad=14,
    )
    lines = [line_auprc, line_auroc, line_loss]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] training_curves.png  (best_epoch={best_epoch}, val_auprc={best_auprc:.3f})")


# ---------------------------------------------------------------------------
# 2. Precision-recall curve  (copy real eval output)
# ---------------------------------------------------------------------------

def copy_pr_curve() -> None:
    src = DATA / "models" / "ensemble" / "eval" / "test_pr.png"
    dst = OUT / "pr_curves.png"
    shutil.copyfile(src, dst)
    print(f"[ok] pr_curves.png        (copied real eval plot from {src.name})")


# ---------------------------------------------------------------------------
# 3. System architecture diagram
# ---------------------------------------------------------------------------

def make_architecture() -> None:
    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=160, facecolor="white")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    def box(x, y, w, h, title, subtitle="", color=PALETTE["panel"], text_color="white", border=None):
        border = border or PALETTE["border"]
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            facecolor=color, edgecolor=border, linewidth=1.8,
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                    fontsize=11, color=text_color, fontweight="bold")
            ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
                    fontsize=8.5, color=text_color, alpha=0.85)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=11, color=text_color, fontweight="bold")

    def arrow(x1, y1, x2, y2, color=PALETTE["border"], style="-|>"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=15, color=color, linewidth=1.6,
            shrinkA=2, shrinkB=2,
        ))

    # Title
    ax.text(7.5, 7.0, "Meshwatch system architecture",
            fontsize=15, fontweight="bold", color=PALETTE["bg"], ha="center")
    ax.text(7.5, 6.65, "Data flows left to right. End-to-end latency from Kafka event to dashboard alert: ~50 ms.",
            fontsize=9.5, color=PALETTE["panel_alt"], ha="center", style="italic")

    # Row 1: pipeline (y ~ 4.0-5.2)
    box(0.2, 4.0, 2.3, 1.2, "Kafka topic", "transactions stream", PALETTE["amber"])
    arrow(2.5, 4.6, 3.1, 4.6)
    box(3.1, 4.0, 2.4, 1.2, "API workers", "FastAPI + Ray Serve", PALETTE["blue"])
    arrow(5.5, 4.6, 6.1, 4.6)
    box(6.1, 4.0, 2.8, 1.2, "Ensemble model", "HeteroGNN + XGBoost", PALETTE["red"])
    arrow(8.9, 4.6, 9.5, 4.6)
    box(9.5, 4.0, 2.5, 1.2, "Predictions buffer", "in-memory ring", PALETTE["panel"])
    arrow(12.0, 4.6, 12.6, 4.6)
    box(12.6, 4.0, 2.2, 1.2, "WebSocket", "/ws/alerts", PALETTE["cyan"])

    # Row 2: consumers (y ~ 1.2-2.4)
    box(0.2, 1.2, 3.4, 1.4, "Drift detector", "PSI / KS / JSD per feature", PALETTE["purple"])
    box(4.2, 1.2, 3.4, 1.4, "AI investigator", "LangGraph agent + 8 tools", PALETTE["green"])
    box(8.2, 1.2, 3.4, 1.4, "Prometheus + Grafana", "latency, drift, alert volume", PALETTE["panel_alt"])
    box(12.0, 1.2, 2.8, 1.4, "React dashboard", "live alerts + cases", PALETTE["red_light"])

    # Cross-row arrows
    arrow(7.5, 4.0, 4.2 + 1.7, 2.6, color=PALETTE["green"])      # ensemble -> investigator
    arrow(7.5, 4.0, 8.2 + 1.7, 2.6, color=PALETTE["panel_alt"])  # ensemble -> prometheus
    arrow(3.1 + 1.2, 4.0, 0.2 + 1.7, 2.6, color=PALETTE["purple"])  # API -> drift
    arrow(13.7, 4.0, 13.4, 2.6, color=PALETTE["red_light"])      # WS -> dashboard
    arrow(7.6, 1.9, 12.0, 1.9, color=PALETTE["green"])           # investigator -> dashboard

    # Storage tray under everything
    fig.tight_layout()
    fig.savefig(OUT / "architecture.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[ok] architecture.png")


# ---------------------------------------------------------------------------
# 4. Heterogeneous graph schema diagram
# ---------------------------------------------------------------------------

def make_graph_schema() -> None:
    fig, ax = plt.subplots(figsize=(11, 9), dpi=160, facecolor="white")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title
    ax.text(5.5, 8.55, "Meshwatch heterogeneous graph schema",
            fontsize=15, fontweight="bold", color=PALETTE["bg"], ha="center")
    ax.text(5.5, 8.2, "5 node types, 6 edge types, ~700k nodes, ~3M edges on the full IEEE-CIS dataset.",
            fontsize=10, color=PALETTE["panel_alt"], ha="center", style="italic")

    nodes = {
        "Transaction": dict(xy=(5.5, 4.8), color=PALETTE["red"], r=0.95, subtitle="scored entity"),
        "Card":        dict(xy=(1.8, 6.4), color=PALETTE["blue"], r=0.75, subtitle="initiator"),
        "Merchant":    dict(xy=(9.2, 6.4), color=PALETTE["green"], r=0.75, subtitle="target"),
        "Device":      dict(xy=(1.8, 2.6), color=PALETTE["purple"], r=0.75, subtitle="origin"),
        "Address":     dict(xy=(9.2, 2.6), color=PALETTE["amber"], r=0.75, subtitle="billing"),
    }

    # Edges: (src, dst, label, label_offset, color)
    edges = [
        ("Transaction", "Card",     "txn_uses_card",     ( -1.8,  0.55), PALETTE["blue"]),
        ("Transaction", "Merchant", "txn_at_merchant",   (  1.8,  0.55), PALETTE["green"]),
        ("Transaction", "Device",   "txn_on_device",     ( -1.8, -0.55), PALETTE["purple"]),
        ("Transaction", "Address",  "txn_at_address",    (  1.8, -0.55), PALETTE["amber"]),
    ]

    # Draw card-card lateral edges first (so they sit behind nodes)
    ax.add_patch(FancyArrowPatch(
        (1.55, 6.0), (1.55, 3.0),
        arrowstyle="-", connectionstyle="arc3,rad=-0.55",
        color="#94a3b8", linestyle=":", linewidth=1.4,
    ))
    ax.text(0.35, 4.5, "card_used_device\n(aggregated)", ha="left", va="center",
            fontsize=8.5, color="#64748b", style="italic")

    ax.add_patch(FancyArrowPatch(
        (8.6, 6.0), (8.6, 3.0),
        arrowstyle="-", connectionstyle="arc3,rad=0.55",
        color="#94a3b8", linestyle=":", linewidth=1.4,
    ))
    ax.text(10.65, 4.5, "card_at_merchant\n(aggregated)", ha="right", va="center",
            fontsize=8.5, color="#64748b", style="italic")

    # Draw main edges
    for src, dst, label, (lx, ly), color in edges:
        x1, y1 = nodes[src]["xy"]
        x2, y2 = nodes[dst]["xy"]
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=14, color=color, linewidth=2,
            shrinkA=nodes[src]["r"] * 70, shrinkB=nodes[dst]["r"] * 70,
        ))
        mx, my = (x1 + x2) / 2 + lx * 0.0, (y1 + y2) / 2 + ly * 0.0
        # Use the offset to nudge label away from the line
        ax.text((x1 + x2) / 2 + lx * 0.3, (y1 + y2) / 2 + ly * 0.3, label,
                ha="center", va="center", fontsize=9, color=color, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))

    # Draw nodes on top
    for name, props in nodes.items():
        x, y = props["xy"]
        r = props["r"]
        c = Circle((x, y), r, facecolor=props["color"], edgecolor=PALETTE["bg"], linewidth=2.4)
        ax.add_patch(c)
        ax.text(x, y + 0.10, name, ha="center", va="center",
                fontsize=11.5, color="white", fontweight="bold")
        ax.text(x, y - 0.22, props["subtitle"], ha="center", va="center",
                fontsize=8, color="white", alpha=0.85, style="italic")

    # Legend strip at bottom
    ax.text(5.5, 0.55,
            "All edges built strictly temporally: only history visible at scoring time. No future leakage by construction.",
            fontsize=9.5, color=PALETTE["panel_alt"], ha="center", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "graph_schema.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[ok] graph_schema.png")


if __name__ == "__main__":
    make_training_curves()
    copy_pr_curve()
    make_architecture()
    make_graph_schema()
    print(f"\nAll images written to {OUT}")
