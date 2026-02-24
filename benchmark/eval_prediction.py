"""Evaluate DFM prediction across context sizes and forecast horizons."""

import argparse
import csv
import json
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from sklearn.metrics import roc_auc_score

from dfm_server.client import DFMClient

HORIZON_BINS = [
    (1, 5),
    (6, 10),
    (11, 25),
    (26, 50),
    (51, 100),
    (101, 200),
    (201, 500),
]


def load_trajectories(path: str) -> list[dict]:
    trajs = []
    with open(path) as f:
        for line in f:
            trajs.append(json.loads(line))
    return trajs


def compute_metrics(preds: np.ndarray, truths: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - truths)))
    acc = float(np.mean((preds >= 0.5) == (truths >= 0.5)))
    labels = (truths >= 0.5).astype(int)
    if len(np.unique(labels)) == 2:
        auc = float(roc_auc_score(labels, preds))
    else:
        auc = None
    return {"mae": mae, "acc": acc, "auc": auc}


def compute_metrics_per_traj(
    preds: np.ndarray,
    truths: np.ndarray,
    sources: np.ndarray,
) -> dict[str, tuple[float, float] | tuple[None, None]]:
    """Compute metrics per trajectory, return (mean, std) across trajectories."""
    unique_sources = np.unique(sources)
    maes, accs, aucs = [], [], []
    for s in unique_sources:
        m = sources == s
        metrics = compute_metrics(preds[m], truths[m])
        maes.append(metrics["mae"])
        accs.append(metrics["acc"])
        if metrics["auc"] is not None:
            aucs.append(metrics["auc"])
    return {
        "mae": (float(np.mean(maes)), float(np.std(maes))),
        "acc": (float(np.mean(accs)), float(np.std(accs))),
        "auc": (float(np.mean(aucs)), float(np.std(aucs))) if aucs else (None, None),
    }


def fmt(val, f=".4f") -> str:
    return "N/A" if val is None else f"{val:{f}}"


def fmt_std(mean, std, f=".4f") -> str:
    return "N/A" if mean is None else f"{mean:{f}}\u00b1{std:{f}}"


def build_tables(
    results: list[dict],
    context_sizes: list[int],
    min_samples: int,
) -> tuple[Table, Table]:
    if not results:
        empty = Table(title="No data yet")
        return empty, empty

    arr = np.array(
        [(r["K"], r["h"], r["pred"], r["truth"]) for r in results], dtype=np.float64
    )
    sources = np.array([r["source"] for r in results])
    ks, hs, preds, truths = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    def make_row(mask):
        m = compute_metrics_per_traj(preds[mask], truths[mask], sources[mask])
        return [fmt_std(*m["mae"]), fmt_std(*m["acc"]), fmt_std(*m["auc"])]

    # Table 1: by context size
    t1 = Table(title="By Context Size (K)")
    for col in ["K", "MAE", "Acc", "AUC", "n_traj", "n"]:
        t1.add_column(col, justify="right")

    for k in sorted(context_sizes):
        mask = ks == k
        n_traj = len(np.unique(sources[mask])) if mask.any() else 0
        if n_traj < min_samples:
            continue
        t1.add_row(str(int(k)), *make_row(mask), str(n_traj), str(int(mask.sum())))

    # Table 2: by horizon bin
    t2 = Table(title="By Horizon Bin (h)")
    for col in ["Horizon", "MAE", "Acc", "AUC", "n_traj", "n"]:
        t2.add_column(col, justify="right")

    for lo, hi in HORIZON_BINS:
        mask = (hs >= lo) & (hs <= hi)
        n_traj = len(np.unique(sources[mask])) if mask.any() else 0
        if n_traj < min_samples:
            continue
        t2.add_row(f"{lo}-{hi}", *make_row(mask), str(n_traj), str(int(mask.sum())))

    return t1, t2


METRIC_NAMES = ["MAE", "Acc", "AUC"]
CMAPS = {"MAE": "RdYlGn_r", "Acc": "RdYlGn", "AUC": "RdYlGn"}
VRANGES = {"MAE": (0, 0.5), "Acc": (0.5, 1.0), "AUC": (0.5, 1.0)}


def compute_matrices(
    results: list[dict],
    context_sizes: list[int],
    min_samples: int,
) -> dict[str, np.ndarray]:
    sorted_ks = sorted(context_sizes)
    n_rows, n_cols = len(sorted_ks), len(HORIZON_BINS)
    matrices = {name: np.full((n_rows, n_cols), np.nan) for name in METRIC_NAMES}

    if not results:
        return matrices

    arr = np.array(
        [(r["K"], r["h"], r["pred"], r["truth"]) for r in results], dtype=np.float64
    )
    sources = np.array([r["source"] for r in results])
    ks_arr, hs_arr, preds, truths = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    for i, k in enumerate(sorted_ks):
        for j, (lo, hi) in enumerate(HORIZON_BINS):
            mask = (ks_arr == k) & (hs_arr >= lo) & (hs_arr <= hi)
            n_traj = len(np.unique(sources[mask])) if mask.any() else 0
            if n_traj < min_samples:
                continue
            m = compute_metrics(preds[mask], truths[mask])
            matrices["MAE"][i, j] = m["mae"]
            matrices["Acc"][i, j] = m["acc"]
            if m["auc"] is not None:
                matrices["AUC"][i, j] = m["auc"]

    return matrices


def init_plot(context_sizes: list[int], interactive: bool = True) -> tuple:
    if interactive:
        plt.ion()
    else:
        plt.ioff()
    sorted_ks = sorted(context_sizes)
    n_rows, n_cols = len(sorted_ks), len(HORIZON_BINS)
    h_labels = [f"{lo}-{hi}" for lo, hi in HORIZON_BINS]
    k_labels = [str(k) for k in sorted_ks]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    images, texts = {}, {}

    for ax, name in zip(axes, METRIC_NAMES):
        vmin, vmax = VRANGES[name]
        im = ax.imshow(
            np.full((n_rows, n_cols), np.nan),
            aspect="auto",
            cmap=CMAPS[name],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(h_labels, rotation=45, ha="right")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(k_labels)
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel("Context size (K)")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.8)
        images[name] = im

        cell_texts = {}
        for i in range(n_rows):
            for j in range(n_cols):
                cell_texts[(i, j)] = ax.text(
                    j, i, "-", ha="center", va="center", color="gray", fontsize=8
                )
        texts[name] = cell_texts

    fig.suptitle("Prediction Evaluation: K \u00d7 Horizon", fontweight="bold")
    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    return fig, axes, images, texts


def update_plot(fig, images, texts, matrices):
    n_rows, n_cols = matrices["MAE"].shape
    for name in METRIC_NAMES:
        mat = matrices[name]
        vmin, vmax = VRANGES[name]
        images[name].set_data(mat)
        for i in range(n_rows):
            for j in range(n_cols):
                val = mat[i, j]
                t = texts[name][(i, j)]
                if np.isnan(val):
                    t.set_text("-")
                    t.set_color("gray")
                else:
                    t.set_text(f"{val:.3f}")
                    t.set_color(
                        "white" if abs(val - vmin) > 0.6 * (vmax - vmin) else "black"
                    )
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def build_display(results, context_sizes, min_samples, progress_text) -> Table:
    t1, t2 = build_tables(results, context_sizes, min_samples)
    layout = Table.grid(padding=1)
    layout.add_row(Text(progress_text, style="bold cyan"))
    layout.add_row(t1)
    layout.add_row(t2)
    return layout


def run(args):
    client = DFMClient(args.server_url)
    all_trajectories = load_trajectories(args.data)
    if args.filter:
        all_trajectories = [
            t for t in all_trajectories
            if any(k in t.get("source", "") for k in args.filter)
        ]
    context_sizes = args.context_sizes
    max_horizon = args.max_horizon
    min_samples = args.min_samples

    min_length = max(context_sizes) + max_horizon
    trajectories = [t for t in all_trajectories if len(t["tasks"]) >= min_length]

    console = Console()
    console.print(
        f"Filtered trajectories: {len(trajectories)}/{len(all_trajectories)} "
        f"(require len >= {min_length})"
    )
    if not trajectories:
        console.print(
            "[red]No trajectories meet the minimum length requirement. "
            "Try reducing --max-horizon or --context-sizes.[/red]"
        )
        return

    show_plot = not args.save
    results: list[dict] = []

    if show_plot:
        fig, axes, images, texts = init_plot(context_sizes)

    rng = np.random.default_rng(seed=42)

    max_K = max(context_sizes)

    with Live(console=console, refresh_per_second=2) as live:
        for ti, traj in enumerate(trajectories):
            all_tasks = traj["tasks"]
            all_outcomes = traj["outcomes"]
            all_answers = traj.get("answers")
            source = traj.get("source", "unknown")
            traj_id = f"{source}_{ti}"

            # Pick a random split point p with room for max_K before and max_horizon after
            lo = max_K
            hi = len(all_tasks) - max_horizon
            p = int(rng.integers(lo, hi + 1))

            # Future tasks are the same for all K values
            fut_tasks = all_tasks[p : p + max_horizon]
            fut_outcomes = all_outcomes[p : p + max_horizon]

            for K in context_sizes:
                progress = (
                    f"Trajectory {ti + 1}/{len(trajectories)} | "
                    f"context={K} | split={p}"
                )
                live.update(
                    build_display(results, context_sizes, min_samples, progress)
                )

                learner_id = str(uuid.uuid4())

                try:
                    # Register with K history steps ending at the split point
                    if K > 0:
                        ctx_tasks = all_tasks[p - K : p]
                        ctx_outcomes = all_outcomes[p - K : p]
                        ctx_answers = (
                            all_answers[p - K : p] if all_answers else None
                        )
                        client.register(
                            learner_id,
                            tasks=ctx_tasks,
                            outcomes=ctx_outcomes,
                            answers=ctx_answers,
                        )
                    else:
                        client.register(learner_id)

                    # Predict future tasks (single deterministic forward pass)
                    preds = client.predict(learner_id, curriculum=[fut_tasks])[0]
                finally:
                    try:
                        client.delete(learner_id)
                    except Exception:
                        pass

                # Collect results
                for h_idx, (pred, truth) in enumerate(
                    zip(preds, fut_outcomes)
                ):
                    results.append(
                        {
                            "K": K,
                            "h": h_idx + 1,
                            "pred": pred,
                            "truth": truth,
                            "source": traj_id,
                        }
                    )

                live.update(
                    build_display(results, context_sizes, min_samples, progress)
                )
                if show_plot:
                    matrices = compute_matrices(results, context_sizes, min_samples)
                    update_plot(fig, images, texts, matrices)

    # Final display
    console.print()
    t1, t2 = build_tables(results, context_sizes, min_samples)
    console.print(t1)
    console.print(t2)
    console.print(f"\nTotal samples: {len(results)}")

    if args.save:
        save_path = Path(args.save)

        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["K", "h", "pred", "truth", "source"])
            writer.writeheader()
            writer.writerows(results)
        console.print(f"Saved {len(results)} rows to {save_path}")

        fig, axes, images, texts = init_plot(context_sizes, interactive=False)
        matrices = compute_matrices(results, context_sizes, min_samples)
        update_plot(fig, images, texts, matrices)
        fig_path = save_path.with_suffix(".png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.print(f"Saved plot to {fig_path}")
    else:
        plt.ioff()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DFM prediction across context sizes and horizons"
    )
    parser.add_argument("--data", required=True, help="Path to val.jsonl")
    parser.add_argument(
        "--server-url", default="http://localhost:8000", help="DFM server URL"
    )
    parser.add_argument(
        "--context-sizes",
        type=int,
        nargs="+",
        default=[0, 10, 25, 50, 100, 200, 500],
        help="Context sizes to evaluate",
    )
    parser.add_argument(
        "--max-horizon", type=int, default=500, help="Maximum forecast horizon"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum trajectories per cell to report",
    )
    parser.add_argument("--save", type=str, help="Save raw results to CSV (+ PNG)")
    parser.add_argument(
        "--filter",
        type=str,
        nargs="+",
        help="Only use trajectories whose source contains any of these keywords",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
