"""
Generate CartPole comparison figures from local CSV and visit-count files.

Usage (after all 4 runs are done):
    python -m scripts.plot_cartpole

Outputs:
    results/cartpole_learning_curves.png  — success rate + EWMA + return, all 4 methods
    results/cartpole_heatmaps.png         — side-by-side state visitation heatmaps
"""

import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure

METHODS = [
    ("entropy",  "Entropy Reg",  "checkpoints/ppo_cartpole_sparse_entropy"),
    ("count",    "Count-based",  "checkpoints/ppo_cartpole_sparse_count"),
    ("icm",      "ICM",          "checkpoints/ppo_cartpole_sparse_icm"),
    ("rnd",      "RND",          "checkpoints/ppo_cartpole_sparse_rnd"),
]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

os.makedirs("results", exist_ok=True)


def load_csv(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot_learning_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].set_title("Success Rate")
    axes[1].set_title("EWMA Success Rate")
    axes[2].set_title("Average Return")

    for ax in axes:
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("")
        ax.grid(True, alpha=0.3)

    any_plotted = False
    for (key, label, base_path), color in zip(METHODS, COLORS):
        csv_path = f"{base_path}_eval.csv"
        data = load_csv(csv_path)
        if data is None:
            print(f"[SKIP] {csv_path} not found — run not complete yet")
            continue

        steps = [d["timestep"] for d in data]
        axes[0].plot(steps, [d["success_rate"] for d in data],
                     label=label, color=color, linewidth=2)
        axes[1].plot(steps, [d["ewma_success_rate"] for d in data],
                     label=label, color=color, linewidth=2)
        axes[2].plot(steps, [d["avg_return"] for d in data],
                     label=label, color=color, linewidth=2)
        any_plotted = True

    if not any_plotted:
        print("[ERROR] No CSV files found. Run training first.")
        return

    # Draw the 0.8 threshold line on success rate plots
    for ax in axes[:2]:
        ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, label="80% threshold")
        ax.set_ylim(0, 1.05)

    axes[0].legend(fontsize=8)
    fig.suptitle("Sparse CartPole — Exploration Method Comparison", fontsize=13, y=1.01)
    fig.tight_layout()

    out = "results/cartpole_learning_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] Learning curves saved to {out}")
    plt.close(fig)


def plot_heatmaps():
    available = []
    for key, label, base_path in METHODS:
        npy_path = f"{base_path}_visits.npy"
        if os.path.exists(npy_path):
            visit_counts = np.load(npy_path, allow_pickle=True).item()
            available.append((label, visit_counts))
        else:
            print(f"[SKIP] {npy_path} not found — run not complete yet")

    if not available:
        print("[ERROR] No visit count files found. Run training first.")
        return

    n = len(available)
    # 5 inches per panel both wide and tall → each panel is square
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (label, visit_counts) in zip(axes, available):
        pos_bins = sorted(set(k[0] for k in visit_counts))
        angle_bins = sorted(set(k[1] for k in visit_counts))

        pos_idx = {v: i for i, v in enumerate(pos_bins)}
        angle_idx = {v: i for i, v in enumerate(angle_bins)}

        grid = np.zeros((len(angle_bins), len(pos_bins)))
        for (pos, angle), count in visit_counts.items():
            grid[angle_idx[angle], pos_idx[pos]] = count

        # aspect="auto" stretches cells to fill the square panel
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        fig.colorbar(im, ax=ax, label="Visit Count")

        # x-axis ticks
        step = max(1, len(pos_bins) // 8)
        x_ticks = list(range(0, len(pos_bins), step))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{pos_bins[i]:.1f}" for i in x_ticks], rotation=45, fontsize=7)

        # y-axis ticks
        ax.set_yticks(range(len(angle_bins)))
        ax.set_yticklabels([f"{v:.1f}" for v in angle_bins], fontsize=7)

        # target x=1.5 marker
        target_col = min(pos_idx, key=lambda p: abs(p - 1.5))
        ax.axvline(x=pos_idx[target_col], color="cyan",
                   linestyle="--", linewidth=1.5, label="target x≈1.5")

        ax.set_xlabel("Cart Position")
        ax.set_ylabel("Pole Angle (rad)")
        ax.set_title(f"{label} CartPole\nVisit Count")
        ax.legend(fontsize=7)

    fig.suptitle("Sparse CartPole — State Visitation Heatmaps", fontsize=13, y=1.01)
    fig.tight_layout()

    out = "results/cartpole_heatmaps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] Heatmaps saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_learning_curves()
    plot_heatmaps()
