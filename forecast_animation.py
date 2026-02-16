"""Animate DFM forecast rollouts against ground truth."""

import argparse
import json
import random
import uuid

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from dfm_server.client import DFMClient


def running_mean(x, w):
    """Causal running mean with edge handling."""
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - w + 1)
        out[i] = np.mean(x[start : i + 1])
    return out


def load_trajectory(path, seed):
    """Load a random trajectory from val.jsonl."""
    with open(path) as f:
        lines = f.readlines()
    rng = random.Random(seed)
    line = rng.choice(lines)
    record = json.loads(line)
    return record["source"], record["tasks"], record["outcomes"]


def main():
    parser = argparse.ArgumentParser(description="Animate DFM forecast rollouts")
    parser.add_argument("--data", default="checkpoints/dfm_training/val.jsonl")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--n-forecasts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoothing-window", type=int, default=10)
    parser.add_argument("--save", default=None, help="Save to file instead of showing")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load trajectory
    source, tasks, outcomes = load_trajectory(args.data, args.seed)
    outcomes = [float(o) for o in outcomes]
    n = len(tasks)
    print(f"Trajectory: {source} ({n} steps)")

    # Split point: between 20% and 60%
    split = rng.randint(int(n * 0.2), int(n * 0.6))
    hist_tasks, fut_tasks = tasks[:split], tasks[split:]
    hist_outcomes, fut_outcomes = outcomes[:split], outcomes[split:]
    n_fut = len(fut_tasks)
    print(f"Split at {split}: {split} history, {n_fut} future")

    # Register learner with history, get forecasts, then clean up
    client = DFMClient(args.server_url)
    learner_id = f"anim-{uuid.uuid4().hex[:8]}"
    try:
        client.register(learner_id, tasks=hist_tasks, outcomes=hist_outcomes)
        # N copies of the same future task sequence → N independent stochastic rollouts
        task_sequences = [fut_tasks] * args.n_forecasts
        forecasts = client.forecast(learner_id, task_sequences)
    finally:
        try:
            client.delete(learner_id)
        except Exception:
            pass

    forecasts = [np.array(f) for f in forecasts]
    print(f"Got {len(forecasts)} forecast rollouts of length {n_fut}")

    # Precompute smoothed curves — future is a continuation of history
    w = args.smoothing_window
    hist_arr = np.array(hist_outcomes)
    hist_smooth = running_mean(hist_arr, w)
    # Smooth GT future using history tail as context
    full_gt = np.concatenate([hist_arr, np.array(fut_outcomes)])
    fut_smooth = running_mean(full_gt, w)[split:]
    # Precompute smoothed forecasts (each continues from history)
    forecasts_smooth = []
    for fc in forecasts:
        full_fc = np.concatenate([hist_arr, fc])
        forecasts_smooth.append(running_mean(full_fc, w)[split:])

    # Full x-axes
    x_hist = np.arange(split)
    x_fut = np.arange(split, n)

    # --- Precompute per-k metrics (forecast mean vs raw ground truth) ---
    fut_outcomes_arr = np.array(fut_outcomes)
    fut_binary = (fut_outcomes_arr >= 0.5).astype(float)
    # Check if AUC is computable (need both classes)
    auc_possible = len(np.unique(fut_binary)) > 1

    metrics_by_k = {}  # k -> (acc, auc, mae)
    for k in range(1, args.n_forecasts + 1):
        # Mean predicted probability across k rollouts (raw, not smoothed)
        mean_pred = np.stack(forecasts[:k]).mean(axis=0)
        pred_binary = (mean_pred >= 0.5).astype(float)
        acc = np.mean(pred_binary == fut_binary)
        mae = np.mean(np.abs(mean_pred - fut_outcomes_arr))
        auc = roc_auc_score(fut_binary, mean_pred) if auc_possible else float("nan")
        metrics_by_k[k] = (acc, auc, mae)

    # --- Animation setup ---
    plt.style.use("dark_background")
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax = fig.add_subplot(gs[0])
    ax_metrics = fig.add_subplot(gs[1])
    cmap = plt.cm.cool

    # Palette — saturated enough to pop on dark bg, distinct hues
    COL_HIST = "#5ec4e8"      # bright sky blue
    COL_TRUTH = "#f5a623"     # warm amber/orange
    COL_MEAN = "#bb86fc"      # vivid purple (Material You accent)
    COL_BAND = "#bb86fc"
    COL_SPLIT = "#5c5c7a"
    COL_TEXT = "#d0d0e0"
    COL_BG = "#1a1a2e"
    COL_CARD = "#2a2a45"

    def draw_metrics(k):
        ax_metrics.clear()
        ax_metrics.set_facecolor(COL_BG)
        ax_metrics.set_xlim(0, 1)
        ax_metrics.set_ylim(0, 1)
        ax_metrics.axis("off")

        if k == 0:
            ax_metrics.text(0.5, 0.5, "Awaiting\nforecasts...",
                            transform=ax_metrics.transAxes, ha="center", va="center",
                            fontsize=11, color="#666680", style="italic")
            return

        acc, auc, mae = metrics_by_k[k]

        # Metric cards
        metrics = [
            ("Acc", acc, "#66d9a0"),   # green
            ("AUC", auc, "#5ec4e8"),   # blue
            ("MAE", mae, "#f5a623"),   # orange
        ]

        y_positions = [0.78, 0.50, 0.22]
        for (name, val, color), y in zip(metrics, y_positions):
            # Card background
            card = plt.matplotlib.patches.FancyBboxPatch(
                (0.08, y - 0.10), 0.84, 0.20,
                transform=ax_metrics.transAxes, clip_on=False,
                facecolor=COL_CARD, edgecolor="#444466",
                linewidth=1, zorder=1, alpha=0.9,
                boxstyle="round,pad=0.02")
            ax_metrics.add_patch(card)

            # Label
            ax_metrics.text(0.5, y + 0.04, name,
                            transform=ax_metrics.transAxes, ha="center", va="center",
                            fontsize=10, color="#888899", fontweight="medium", zorder=2)
            # Value
            val_str = f"{val:.3f}" if not np.isnan(val) else "N/A"
            ax_metrics.text(0.5, y - 0.04, val_str,
                            transform=ax_metrics.transAxes, ha="center", va="center",
                            fontsize=16, color=color, fontweight="bold", zorder=2,
                            fontfamily="monospace")

    def draw_frame(frame_idx):
        ax.clear()
        ax.set_facecolor(COL_BG)

        # History: smoothed running mean
        ax.plot(x_hist, hist_smooth, color=COL_HIST, linewidth=2.2,
                label="History", zorder=3)

        # Ground truth future
        ax.plot(x_fut, fut_smooth, color=COL_TRUTH, linewidth=2.2, linestyle="-",
                alpha=0.85, label="Ground truth", zorder=4)

        # Vertical split line
        ax.axvline(split, color=COL_SPLIT, linestyle="--", linewidth=1.2,
                   alpha=0.6, zorder=1)

        # Forecast trajectories revealed so far
        k = frame_idx  # number of forecasts shown (0 on first frame)
        for i in range(k):
            t = i / max(args.n_forecasts - 1, 1)
            color = cmap(t)
            ax.plot(x_fut, forecasts_smooth[i], color=color, linewidth=0.8,
                    alpha=0.35, zorder=2)

        # Running mean + confidence band
        if k > 0:
            stack = np.stack(forecasts_smooth[:k])
            mean_line = stack.mean(axis=0)
            std_line = stack.std(axis=0)

            ax.fill_between(x_fut, mean_line - std_line, mean_line + std_line,
                            color=COL_BAND, alpha=0.12, zorder=2)
            ax.plot(x_fut, mean_line, color=COL_MEAN, linewidth=2.5, alpha=0.95,
                    label=f"Mean ({k} forecasts)", zorder=5)

        # Styling
        ax.set_xlim(-1, n + 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Step", fontsize=11, color=COL_TEXT)
        ax.set_ylabel("Outcome", fontsize=11, color=COL_TEXT)
        ax.set_title(source, fontsize=14, fontweight="bold", color="#e8e8f0",
                     pad=12)
        ax.tick_params(colors="#888899", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#333350")

        ax.text(0.98, 0.95, f"{k}/{args.n_forecasts} forecasts",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, color=COL_TEXT, fontweight="medium",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=COL_CARD,
                          edgecolor="#444466", alpha=0.85))
        ax.legend(loc="lower left", fontsize=9, framealpha=0.5,
                  facecolor=COL_CARD, edgecolor="#444466",
                  labelcolor=COL_TEXT)

        # Metrics side panel
        draw_metrics(k)

    # Frames: 0 (history only) through n_forecasts (all shown)
    n_frames = args.n_forecasts + 1
    anim = animation.FuncAnimation(fig, draw_frame, frames=n_frames,
                                   interval=500, repeat=True)

    fig.tight_layout(rect=[0.03, 0.05, 1, 0.95])

    if args.save:
        anim.save(args.save, writer="pillow", fps=2)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
