"""Visualize DFM prediction vs ground truth on a single trajectory."""

import argparse
import json
import random
import uuid

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
    return record["source"], record["tasks"], record["outcomes"], record.get("answers")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DFM prediction vs ground truth"
    )
    parser.add_argument("--data", default="checkpoints/val.jsonl")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoothing-window", type=int, default=10)
    parser.add_argument(
        "--max-context", type=int, default=300, help="Max history steps as context"
    )
    parser.add_argument(
        "--max-future", type=int, default=300, help="Max future steps to predict"
    )
    parser.add_argument("--save", default=None, help="Save to file instead of showing")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load trajectory
    source, tasks, outcomes, answers = load_trajectory(args.data, args.seed)
    outcomes = [float(o) for o in outcomes]
    n = len(tasks)
    print(f"Trajectory: {source} ({n} steps)")

    # Split: between 20% and 60%
    split = rng.randint(int(n * 0.2), int(n * 0.6))
    # Full history for plotting
    full_hist_outcomes = outcomes[:split]
    # Truncated history for server context
    ctx_tasks = tasks[:split][-args.max_context :]
    ctx_outcomes = outcomes[:split][-args.max_context :]
    ctx_answers = (answers[:split][-args.max_context :]) if answers else None
    n_ctx = len(ctx_tasks)
    fut_tasks = tasks[split : split + args.max_future]
    fut_outcomes = outcomes[split : split + args.max_future]
    n_fut = len(fut_tasks)
    print(
        f"Split at {split}: {len(full_hist_outcomes)} full history, "
        f"{n_ctx} context (max {args.max_context}), {n_fut} future (max {args.max_future})"
    )

    # Register learner, predict, clean up
    client = DFMClient(args.server_url)
    learner_id = f"viz-{uuid.uuid4().hex[:8]}"
    try:
        client.register(
            learner_id,
            tasks=ctx_tasks,
            outcomes=ctx_outcomes,
            answers=ctx_answers,
        )
        preds = client.predict(learner_id, curriculum=[fut_tasks])[0]
    finally:
        try:
            client.delete(learner_id)
        except Exception:
            pass

    preds = np.array(preds)
    print(f"Got {len(preds)} predictions")

    # Smoothing
    w = args.smoothing_window
    full_hist_arr = np.array(full_hist_outcomes)
    n_full_hist = len(full_hist_arr)
    full_hist_smooth = running_mean(full_hist_arr, w)

    # Context start index within the full history
    ctx_start = n_full_hist - n_ctx

    full_gt = np.concatenate([full_hist_arr, np.array(fut_outcomes)])
    fut_smooth = running_mean(full_gt, w)[n_full_hist - 1 :]

    full_pred = np.concatenate([full_hist_arr, preds])
    pred_smooth = running_mean(full_pred, w)[n_full_hist - 1 :]

    # Metrics
    fut_arr = np.array(fut_outcomes)
    fut_binary = (fut_arr >= 0.5).astype(float)
    pred_binary = (preds >= 0.5).astype(float)
    acc = float(np.mean(pred_binary == fut_binary))
    mae = float(np.mean(np.abs(preds - fut_arr)))
    auc_possible = len(np.unique(fut_binary)) > 1
    auc = float(roc_auc_score(fut_binary, preds)) if auc_possible else float("nan")

    # Plot
    n_total = n_full_hist + n_fut
    x_hist = np.arange(n_full_hist)
    x_fut = np.arange(n_full_hist - 1, n_total)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    COL_UNUSED = "#2a6f85"
    COL_HIST = "#5ec4e8"
    COL_TRUTH = "#f5a623"
    COL_PRED = "#bb86fc"
    COL_SPLIT = "#5c5c7a"

    # Skip smoothing warm-up region
    warmup = min(w, n_full_hist)

    # Unused early history (dimmer)
    if ctx_start > 0:
        start = max(warmup, 0)
        # +1 to overlap with context segment at the boundary
        end = ctx_start + 1
        if start < end:
            ax.plot(
                x_hist[start:end],
                full_hist_smooth[start:end],
                color=COL_UNUSED,
                linewidth=1.8,
                alpha=0.5,
                label="Unused history",
            )
    # Used context (bright)
    ctx_plot_start = max(ctx_start, warmup)
    ax.plot(
        x_hist[ctx_plot_start:],
        full_hist_smooth[ctx_plot_start:],
        color=COL_HIST,
        linewidth=2.2,
        label=f"Context (K={n_ctx})",
    )
    ax.plot(
        x_fut,
        fut_smooth,
        color=COL_TRUTH,
        linewidth=2.2,
        alpha=0.85,
        label="Ground truth",
    )
    ax.plot(
        x_fut,
        pred_smooth,
        color=COL_PRED,
        linewidth=2.5,
        alpha=0.95,
        label="Prediction",
    )
    # Context start marker
    if ctx_start > 0:
        ax.axvline(
            ctx_start, color=COL_SPLIT, linestyle=":", linewidth=1.0, alpha=0.4
        )
    # Split line
    ax.axvline(
        n_full_hist - 1, color=COL_SPLIT, linestyle="--", linewidth=1.2, alpha=0.6
    )

    ax.set_xlim(warmup - 1, n_total + 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Step", fontsize=11, color="#d0d0e0")
    ax.set_ylabel("Outcome", fontsize=11, color="#d0d0e0")

    metrics_str = f"MAE={mae:.3f}  Acc={acc:.3f}"
    if not np.isnan(auc):
        metrics_str += f"  AUC={auc:.3f}"
    ax.set_title(
        f"{source}\n{metrics_str}",
        fontsize=13,
        fontweight="bold",
        color="#e8e8f0",
        pad=12,
    )

    ax.tick_params(colors="#888899", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333350")
    ax.legend(
        loc="lower left",
        fontsize=9,
        framealpha=0.5,
        facecolor="#2a2a45",
        edgecolor="#444466",
        labelcolor="#d0d0e0",
    )

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
