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
    horizon_bins: list[tuple[int, int]] | None = None,
) -> tuple[Table, Table]:
    if horizon_bins is None:
        horizon_bins = HORIZON_BINS

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

    for lo, hi in horizon_bins:
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
    horizon_bins: list[tuple[int, int]] | None = None,
) -> dict[str, np.ndarray]:
    if horizon_bins is None:
        horizon_bins = HORIZON_BINS
    sorted_ks = sorted(context_sizes)
    n_rows, n_cols = len(sorted_ks), len(horizon_bins)
    matrices = {name: np.full((n_rows, n_cols), np.nan) for name in METRIC_NAMES}

    if not results:
        return matrices

    arr = np.array(
        [(r["K"], r["h"], r["pred"], r["truth"]) for r in results], dtype=np.float64
    )
    sources = np.array([r["source"] for r in results])
    ks_arr, hs_arr, preds, truths = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    for i, k in enumerate(sorted_ks):
        for j, (lo, hi) in enumerate(horizon_bins):
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


def init_plot(
    context_sizes: list[int],
    interactive: bool = True,
    horizon_bins: list[tuple[int, int]] | None = None,
) -> tuple:
    if horizon_bins is None:
        horizon_bins = HORIZON_BINS
    if interactive:
        plt.ion()
    else:
        plt.ioff()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    sorted_ks = sorted(context_sizes)
    n_rows, n_cols = len(sorted_ks), len(horizon_bins)
    h_labels = [f"{lo}\u2013{hi}" for lo, hi in horizon_bins]
    k_labels = [str(k) for k in sorted_ks]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.set_facecolor("white")
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
        ax.set_xticklabels(h_labels, rotation=40, ha="right")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(k_labels)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Context Size (K)")
        ax.set_title(name)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=9, length=0)
        images[name] = im

        cell_texts = {}
        for i in range(n_rows):
            for j in range(n_cols):
                cell_texts[(i, j)] = ax.text(
                    j, i, "", ha="center", va="center",
                    color="#aaaaaa", fontsize=9, fontweight="medium",
                )
        texts[name] = cell_texts

    fig.suptitle(
        "Prediction Evaluation: Context Size \u00d7 Horizon",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    return fig, axes, images, texts


def _text_color(val, vmin, vmax, cmap_name):
    """Pick white or dark text for readability on the cell background."""
    import matplotlib.cm as cm
    norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
    norm = max(0.0, min(1.0, norm))
    r, g, b, _ = cm.get_cmap(cmap_name)(norm)
    # Perceived luminance
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.45 else "#222222"


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
                    t.set_text("")
                    t.set_color("#aaaaaa")
                else:
                    t.set_text(f"{val:.3f}")
                    t.set_color(_text_color(val, vmin, vmax, CMAPS[name]))
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


DIST_BINS = np.linspace(0, 1, 11)  # 10 bins over [0, 1]
DIST_COLOR = "#5A9BD5"


def init_dist_plot(
    context_sizes: list[int],
    interactive: bool = True,
    horizon_bins: list[tuple[int, int]] | None = None,
) -> tuple:
    if horizon_bins is None:
        horizon_bins = HORIZON_BINS
    if interactive:
        plt.ion()
    else:
        plt.ioff()

    sorted_ks = sorted(context_sizes)
    n_rows, n_cols = len(sorted_ks), len(horizon_bins)
    h_labels = [f"{lo}\u2013{hi}" for lo, hi in horizon_bins]
    k_labels = [str(k) for k in sorted_ks]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols, 1.8 * n_rows),
        sharex=True, sharey=True,
        squeeze=False,
    )
    fig.set_facecolor("white")

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7, length=0)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#cccccc")
            if i == n_rows - 1:
                ax.set_xlabel(h_labels[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"K={k_labels[i]}", fontsize=9)
            if i == 0 and j == n_cols // 2:
                ax.set_title("Horizon", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Outcome Distribution per Cell",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0.02, 0, 1, 0.94])
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    return fig, axes


def update_dist_plot(
    fig,
    axes,
    results: list[dict],
    context_sizes: list[int],
    min_samples: int,
    horizon_bins: list[tuple[int, int]] | None = None,
):
    if horizon_bins is None:
        horizon_bins = HORIZON_BINS
    if not results:
        return

    sorted_ks = sorted(context_sizes)
    n_rows, n_cols = len(sorted_ks), len(horizon_bins)

    arr = np.array(
        [(r["K"], r["h"], r["truth"]) for r in results], dtype=np.float64
    )
    ks_arr, hs_arr, truths = arr[:, 0], arr[:, 1], arr[:, 2]
    sources = np.array([r["source"] for r in results])

    cell_freqs = {}
    for i, k in enumerate(sorted_ks):
        for j, (lo, hi) in enumerate(horizon_bins):
            mask = (ks_arr == k) & (hs_arr >= lo) & (hs_arr <= hi)
            n_traj = len(np.unique(sources[mask])) if mask.any() else 0
            if n_traj < min_samples or not mask.any():
                cell_freqs[(i, j)] = None
                continue
            counts, _ = np.histogram(truths[mask], bins=DIST_BINS)
            total = counts.sum()
            cell_freqs[(i, j)] = counts / total if total > 0 else counts

    bin_width = DIST_BINS[1] - DIST_BINS[0]
    bin_centers = (DIST_BINS[:-1] + DIST_BINS[1:]) / 2

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.clear()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.08)
            ax.tick_params(labelsize=7, length=0)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#cccccc")

            freqs = cell_freqs.get((i, j))
            if freqs is not None:
                ax.bar(
                    bin_centers, freqs, width=bin_width * 0.88,
                    color=DIST_COLOR, edgecolor="white", linewidth=0.4,
                )
            else:
                ax.text(
                    0.5, 0.5, "n/a", ha="center", va="center",
                    transform=ax.transAxes, color="#aaaaaa", fontsize=9,
                )

            # Restore labels
            h_labels = [f"{lo}\u2013{hi}" for lo, hi in horizon_bins]
            k_labels = [str(k) for k in sorted_ks]
            if i == n_rows - 1:
                ax.set_xlabel(h_labels[j], fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"K={k_labels[i]}", fontsize=9)
            else:
                ax.set_yticklabels([])

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def build_display(results, context_sizes, min_samples, progress_text, horizon_bins=None) -> Table:
    t1, t2 = build_tables(results, context_sizes, min_samples, horizon_bins)
    layout = Table.grid(padding=1)
    layout.add_row(Text(progress_text, style="bold cyan"))
    layout.add_row(t1)
    layout.add_row(t2)
    return layout


def run(args):
    all_trajectories = load_trajectories(args.data)
    if args.filter:
        all_trajectories = [
            t for t in all_trajectories
            if any(k in t.get("source", "") for k in args.filter)
        ]
    context_sizes = args.context_sizes
    max_horizon = args.max_horizon
    min_samples = args.min_samples
    baseline = args.baseline

    horizon_bins = [(lo, hi) for lo, hi in HORIZON_BINS if lo <= max_horizon]

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

    if baseline:
        mean_outcome = float(np.mean([
            o for t in trajectories for o in t["outcomes"]
        ]))
        console.print(f"[bold]Baseline mode:[/bold] predicting constant {mean_outcome:.4f}")
    else:
        client = DFMClient(args.server_url)

    show_plot = not args.save
    results: list[dict] = []

    if show_plot:
        fig, axes, images, texts = init_plot(context_sizes, horizon_bins=horizon_bins)
        dist_fig, dist_axes = init_dist_plot(context_sizes, horizon_bins=horizon_bins)

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

            if baseline:
                # Batch all K values at once — no server calls
                for K in context_sizes:
                    for h_idx, truth in enumerate(fut_outcomes):
                        results.append(
                            {
                                "K": K,
                                "h": h_idx + 1,
                                "pred": mean_outcome,
                                "truth": truth,
                                "source": traj_id,
                            }
                        )
                progress = (
                    f"[Baseline] Trajectory {ti + 1}/{len(trajectories)} | split={p}"
                )
                live.update(
                    build_display(results, context_sizes, min_samples, progress, horizon_bins)
                )
                if show_plot:
                    matrices = compute_matrices(results, context_sizes, min_samples, horizon_bins)
                    update_plot(fig, images, texts, matrices)
                    update_dist_plot(dist_fig, dist_axes, results, context_sizes, min_samples, horizon_bins)
            else:
                for K in context_sizes:
                    progress = (
                        f"Trajectory {ti + 1}/{len(trajectories)} | "
                        f"context={K} | split={p}"
                    )
                    live.update(
                        build_display(results, context_sizes, min_samples, progress, horizon_bins)
                    )

                    learner_id = str(uuid.uuid4())
                    try:
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

                        preds = client.predict(learner_id, curriculum=[fut_tasks])[0]
                    finally:
                        try:
                            client.delete(learner_id)
                        except Exception:
                            pass

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
                        build_display(results, context_sizes, min_samples, progress, horizon_bins)
                    )
                    if show_plot:
                        matrices = compute_matrices(results, context_sizes, min_samples, horizon_bins)
                        update_plot(fig, images, texts, matrices)
                        update_dist_plot(dist_fig, dist_axes, results, context_sizes, min_samples, horizon_bins)

    # Final display
    console.print()
    t1, t2 = build_tables(results, context_sizes, min_samples, horizon_bins)
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

        fig, axes, images, texts = init_plot(context_sizes, interactive=False, horizon_bins=horizon_bins)
        matrices = compute_matrices(results, context_sizes, min_samples, horizon_bins)
        update_plot(fig, images, texts, matrices)
        fig_path = save_path.with_suffix(".png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.print(f"Saved plot to {fig_path}")

        dist_fig, dist_axes = init_dist_plot(context_sizes, interactive=False, horizon_bins=horizon_bins)
        update_dist_plot(dist_fig, dist_axes, results, context_sizes, min_samples, horizon_bins)
        dist_path = save_path.with_name(save_path.stem + "_dist.png")
        dist_fig.savefig(dist_path, dpi=150, bbox_inches="tight")
        plt.close(dist_fig)
        console.print(f"Saved distribution plot to {dist_path}")
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
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run mean-outcome baseline instead of the DFM server",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
