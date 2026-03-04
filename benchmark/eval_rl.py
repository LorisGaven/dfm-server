"""Evaluate DFM prediction error as a function of context size and horizon.

Builds a 2D matrix where:
  - Context size (C): number of (task, outcome) pairs given as full history
  - Horizon (H): number of future tasks given as curriculum (task-only, no outcomes)

For each eval step S (position P = S * batch_size in the trajectory), we independently
vary C and H with C + H <= P. This avoids the bias of fixing C + H = P.

Usage:
    python benchmark/eval_rl.py \
        --trajectory /path/to/history.json \
        --eval-dir /path/to/eval_results/ \
        --train-batch-size 64 \
        --save results/
"""

import argparse
import csv
import json
import re
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from dfm_server.client import DFMClient

console = Console()


def load_trajectory(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def discover_eval_files(eval_dir: str) -> list[tuple[int, Path]]:
    """Find step_N.json files, return sorted list of (step, path)."""
    pattern = re.compile(r"^step_(\d+)\.json$")
    results = []
    for p in Path(eval_dir).iterdir():
        m = pattern.match(p.name)
        if m:
            results.append((int(m.group(1)), p))
    results.sort(key=lambda x: x[0])
    return results


def load_eval(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt(val, f=".4f") -> str:
    return "—" if val is None else f"{val:{f}}"


def run(args):
    client = DFMClient(args.server_url)

    # Load trajectory
    traj = load_trajectory(args.trajectory)
    traj_tasks = traj["tasks"]
    traj_outcomes = traj["outcomes"]
    traj_answers = traj.get("answers")
    console.print(f"Loaded trajectory: {len(traj_tasks):,} entries")

    # Discover eval files
    eval_files = discover_eval_files(args.eval_dir)
    if not eval_files:
        console.print("[red]No step_N.json files found in eval dir[/]")
        return
    console.print(
        f"Found {len(eval_files)} eval checkpoints: "
        f"steps {eval_files[0][0]}..{eval_files[-1][0]}"
    )

    # Filter eval steps
    eval_files = [(s, p) for s, p in eval_files if s > args.min_step]
    console.print(
        f"Using {len(eval_files)} eval steps (skipping steps <= {args.min_step})"
    )

    # Verify all eval tasks are in DFM vocabulary
    known_tasks = set(client.tasks())
    console.print(f"DFM vocabulary: {len(known_tasks):,} tasks")
    all_eval_tasks = set()
    for _, eval_path in eval_files:
        eval_data = load_eval(eval_path)
        all_eval_tasks.update(eval_data["tasks"])
    missing = all_eval_tasks - known_tasks
    if missing:
        raise ValueError(
            f"{len(missing)} eval tasks not in DFM vocabulary: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )

    # Build grids
    context_sizes = list(range(0, args.max_context + 1, args.grid_step))
    horizon_sizes = list(range(0, args.max_horizon + 1, args.grid_step))
    console.print(
        f"Grid: {len(context_sizes)} context sizes × {len(horizon_sizes)} horizon sizes "
        f"(step={args.grid_step})"
    )

    # Accumulate results: (C, H) -> list of MAE values across eval steps
    matrix_acc = defaultdict(list)
    raw_results = []  # (context, horizon, step, mae)

    total_eval_steps = len(eval_files)
    for step_idx, (step, eval_path) in enumerate(eval_files):
        P = min(step * args.train_batch_size, len(traj_tasks))
        eval_data = load_eval(eval_path)
        eval_tasks = eval_data["tasks"]
        eval_outcomes = np.array(eval_data["outcomes"])

        n_cells = sum(
            1 for C in context_sizes if C <= P for H in horizon_sizes if C + H <= P
        )
        console.print(
            f"\n[bold]Step {step}[/] (P={P:,}, {n_cells} cells) "
            f"[{step_idx + 1}/{total_eval_steps}]"
        )

        cell_idx = 0
        for C in context_sizes:
            if C > P:
                break
            for H in horizon_sizes:
                if C + H > P:
                    break

                learner_id = str(uuid.uuid4())
                try:
                    # Context: last C entries before the horizon
                    # Horizon: last H entries before the eval point
                    # trajectory[P-C-H : P-H] = context, trajectory[P-H : P] = horizon
                    ctx_start = P - C - H
                    ctx_end = P - H
                    if C > 0:
                        ctx_tasks = traj_tasks[ctx_start:ctx_end]
                        ctx_outcomes = traj_outcomes[ctx_start:ctx_end]
                        ctx_answers = (
                            traj_answers[ctx_start:ctx_end] if traj_answers else None
                        )
                        client.register(
                            learner_id,
                            tasks=ctx_tasks,
                            outcomes=ctx_outcomes,
                            answers=ctx_answers,
                        )
                    else:
                        client.register(learner_id)

                    # Horizon: last H tasks before eval point (task-only, no outcomes)
                    curriculum = [traj_tasks[ctx_end:P]] if H > 0 else None

                    # Predict on eval tasks (batched)
                    preds = []
                    for i in range(0, len(eval_tasks), args.predict_batch_size):
                        batch = eval_tasks[i : i + args.predict_batch_size]
                        batch_preds = client.predict(
                            learner_id,
                            curriculum=curriculum,
                            target_tasks=batch,
                        )
                        preds.extend(batch_preds[0])

                    preds_arr = np.array(preds)
                    mae = float(np.mean(np.abs(preds_arr - eval_outcomes)))

                    matrix_acc[(C, H)].append(mae)
                    raw_results.append((C, H, step, mae))

                    cell_idx += 1
                    console.print(
                        f"  C={C:>5}, H={H:>5} -> MAE={mae:.4f}  ({cell_idx}/{n_cells})"
                    )
                finally:
                    try:
                        client.delete(learner_id)
                    except Exception:
                        pass

    # Build averaged matrix
    console.print("\n[bold]Building matrix...[/]")
    matrix = np.full((len(context_sizes), len(horizon_sizes)), np.nan)
    counts = np.zeros_like(matrix, dtype=int)
    for ci, C in enumerate(context_sizes):
        for hi, H in enumerate(horizon_sizes):
            vals = matrix_acc.get((C, H))
            if vals:
                matrix[ci, hi] = np.mean(vals)
                counts[ci, hi] = len(vals)

    # Display summary table
    table = Table(title="Average MAE (context × horizon)")
    table.add_column("C \\ H", justify="right", style="bold")
    for H in horizon_sizes:
        table.add_column(str(H), justify="right")
    for ci, C in enumerate(context_sizes):
        row = [str(C)]
        for hi in range(len(horizon_sizes)):
            val = matrix[ci, hi]
            row.append(fmt(val) if not np.isnan(val) else "—")
        table.add_row(*row)
    console.print(table)

    # Save
    if args.save and raw_results:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Raw CSV
        csv_path = save_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["context", "horizon", "step", "mae"])
            writer.writeheader()
            for C, H, step, mae in raw_results:
                writer.writerow({"context": C, "horizon": H, "step": step, "mae": mae})
        console.print(f"Saved {len(raw_results)} rows to {csv_path}")

        # Averaged matrix CSV
        matrix_csv_path = save_dir / "matrix.csv"
        with open(matrix_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["context\\horizon"] + [str(H) for H in horizon_sizes])
            for ci, C in enumerate(context_sizes):
                row = [str(C)]
                for hi in range(len(horizon_sizes)):
                    val = matrix[ci, hi]
                    row.append(f"{val:.4f}" if not np.isnan(val) else "")
                writer.writerow(row)
        console.print(f"Saved averaged matrix to {matrix_csv_path}")

        # Heatmap
        _plot_heatmap(matrix, context_sizes, horizon_sizes, save_dir)
        console.print(f"Saved heatmap to {save_dir / 'matrix.png'}")


def _plot_heatmap(matrix, context_sizes, horizon_sizes, save_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    # Mask NaN cells
    masked = np.ma.masked_where(np.isnan(matrix), matrix)

    im = ax.pcolormesh(
        horizon_sizes + [horizon_sizes[-1] + (horizon_sizes[1] - horizon_sizes[0])],
        context_sizes
        + [
            context_sizes[-1] + (context_sizes[1] - context_sizes[0])
            if len(context_sizes) > 1
            else context_sizes[-1] + 256
        ],
        masked,
        cmap="RdYlGn_r",
        shading="flat",
    )

    ax.set_xlabel("Horizon (task-only tokens)", fontsize=12)
    ax.set_ylabel("Context size (full history)", fontsize=12)
    ax.set_title(
        "DFM Prediction MAE: Context × Horizon", fontsize=14, fontweight="bold"
    )

    cbar = fig.colorbar(im, ax=ax, label="MAE")
    cbar.ax.tick_params(labelsize=10)

    # Add text annotations for non-NaN cells
    for ci in range(len(context_sizes)):
        for hi in range(len(horizon_sizes)):
            val = matrix[ci, hi]
            if not np.isnan(val):
                x = (
                    horizon_sizes[hi] + (horizon_sizes[1] - horizon_sizes[0]) / 2
                    if len(horizon_sizes) > 1
                    else horizon_sizes[hi]
                )
                y = (
                    context_sizes[ci] + (context_sizes[1] - context_sizes[0]) / 2
                    if len(context_sizes) > 1
                    else context_sizes[ci]
                )
                ax.text(
                    x,
                    y,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if val > np.nanmedian(matrix) else "black",
                )

    fig.tight_layout()
    fig.savefig(save_dir / "matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DFM prediction error vs context size and horizon"
    )
    parser.add_argument(
        "--trajectory",
        required=True,
        help="Path to training history.json",
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Directory with step_N.json eval files",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        required=True,
        help="RL training batch size (to compute position = step * batch_size)",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="DFM server URL",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=64,
        help="Max eval tasks per predict call",
    )
    parser.add_argument(
        "--grid-step",
        type=int,
        default=512,
        help="Step size for context and horizon grids",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=4096,
        help="Maximum context size",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=4096,
        help="Maximum horizon",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=100,
        help="Skip eval steps <= this value",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Output directory for results (CSV + heatmap)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
