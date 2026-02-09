"""
Benchmark a DFM model checkpoint via the inference server.

Usage:
    # Start the server first:
    DFM_CHECKPOINT_PATH=ckpt.pt DFM_EMBEDDINGS_PATH=embeddings.pt python -m dfm_server.server

    # Then run the benchmark:
    python benchmark.py --data val.jsonl [--server-url http://localhost:8000]
                        [--max-learners 0]
"""

import argparse
import json
import time
from collections import defaultdict

from dfm_server.client import DFMClient

FORECAST_SPLITS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(predictions, targets):
    """Compute BCE, accuracy, MAE, and AUC from lists of floats."""
    import math

    n = len(predictions)
    if n == 0:
        return {"bce": None, "accuracy": None, "mae": None, "auc": None, "n": 0}

    eps = 1e-7
    bce = 0.0
    for p, t in zip(predictions, targets):
        p_c = max(eps, min(1 - eps, p))
        bce += -(t * math.log(p_c) + (1 - t) * math.log(1 - p_c))
    bce /= n

    correct = sum(1 for p, t in zip(predictions, targets) if (p >= 0.5) == (t >= 0.5))
    accuracy = correct / n

    mae = sum(abs(p - t) for p, t in zip(predictions, targets)) / n

    auc = _auc(predictions, targets)

    return {"bce": bce, "accuracy": accuracy, "mae": mae, "auc": auc, "n": n}


def _auc(predictions, targets):
    """Compute AUC-ROC. Returns None if only one class is present."""
    pairs = sorted(zip(predictions, targets), key=lambda x: -x[0])
    n_pos = sum(1 for _, t in pairs if t >= 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    prev_tp = 0
    prev_score = None
    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2
            prev_fp = fp
            prev_tp = tp
        if label >= 0.5:
            tp += 1
        else:
            fp += 1
        prev_score = score
    auc += (fp - prev_fp) * (tp + prev_tp) / 2
    return auc / (n_pos * n_neg)


def calibration_bins(predictions, targets, n_bins=10):
    """Compute calibration: for each probability bin, expected vs observed frequency."""
    bins = defaultdict(lambda: {"pred_sum": 0.0, "target_sum": 0.0, "count": 0})
    for p, t in zip(predictions, targets):
        b = min(int(p * n_bins), n_bins - 1)
        bins[b]["pred_sum"] += p
        bins[b]["target_sum"] += t
        bins[b]["count"] += 1
    result = []
    for b in range(n_bins):
        d = bins[b]
        if d["count"] > 0:
            result.append({
                "bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}",
                "mean_pred": d["pred_sum"] / d["count"],
                "mean_target": d["target_sum"] / d["count"],
                "count": d["count"],
            })
    return result


def fmt_metric(v, fmt=".4f"):
    return f"{v:{fmt}}" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------


def evaluate_prediction(client, learner_id, tasks, outcomes):
    """Next-step prediction: register empty, then predict+update for each step.

    Returns list of (prediction, target, position) tuples.
    """
    client.register(learner_id)
    results = []
    for i, (task, outcome) in enumerate(zip(tasks, outcomes)):
        preds = client.predict(learner_id, [task])
        results.append((preds[0], outcome, i))
        client.update(learner_id, task, outcome)
    client.delete(learner_id)
    return results


def evaluate_forecast(client, learner_id, tasks, outcomes, split_idx):
    """Autoregressive forecast: register with history[:split], forecast the rest.

    Returns list of (prediction, target, horizon) tuples.
    """
    prefix_tasks = tasks[:split_idx]
    prefix_outcomes = outcomes[:split_idx]
    future_tasks = tasks[split_idx:]
    future_outcomes = outcomes[split_idx:]

    if not future_tasks:
        return []

    client.register(learner_id, tasks=prefix_tasks, outcomes=prefix_outcomes)
    preds_list = client.forecast(learner_id, [future_tasks])
    preds = preds_list[0]
    client.delete(learner_id)

    return [(p, t, h) for h, (p, t) in enumerate(zip(preds, future_outcomes))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark DFM model via inference server")
    parser.add_argument("--data", required=True, help="Path to val.jsonl")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--max-learners", type=int, default=0,
                        help="Max learners to evaluate (0 = all)")
    args = parser.parse_args()

    # Check server
    client = DFMClient(args.server_url)
    health = client.health()
    print(f"Server OK: {health['learner_count']} learners registered")

    # Load data
    learners_data = []
    with open(args.data) as f:
        for line in f:
            learners_data.append(json.loads(line))
    if args.max_learners > 0:
        learners_data = learners_data[: args.max_learners]
    print(f"Loaded {len(learners_data)} learners from {args.data}")

    total_steps = sum(len(l["tasks"]) for l in learners_data)
    print(f"Total steps: {total_steps:,}")

    # ===================================================================
    # 1. Next-step prediction (run once)
    # ===================================================================
    print("\n" + "=" * 70)
    print("NEXT-STEP PREDICTION")
    print("=" * 70)

    all_pred_results = []  # (pred, target, position, source)
    per_source_pred = defaultdict(list)
    per_position_pred = defaultdict(list)

    t0 = time.time()
    for i, learner in enumerate(learners_data):
        lid = f"bench_pred_{i}"
        source = learner["source"]
        tasks = learner["tasks"]
        outcomes = learner["outcomes"]
        results = evaluate_prediction(client, lid, tasks, outcomes)
        for pred, target, pos in results:
            all_pred_results.append((pred, target, pos, source))
            per_source_pred[source].append((pred, target, pos))
            per_position_pred[pos].append((pred, target))
        print(f"\r  [{i + 1}/{len(learners_data)}] {source} ({len(tasks)} steps)", end="", flush=True)
    pred_time = time.time() - t0
    print(f"\n  Completed in {pred_time:.1f}s ({total_steps / pred_time:.0f} steps/s)")

    # Overall metrics
    preds_all = [r[0] for r in all_pred_results]
    targets_all = [r[1] for r in all_pred_results]
    overall_pred = compute_metrics(preds_all, targets_all)
    print(f"\n  Overall ({overall_pred['n']:,} predictions):")
    print(f"    BCE:      {fmt_metric(overall_pred['bce'])}")
    print(f"    Accuracy: {fmt_metric(overall_pred['accuracy'])}")
    print(f"    MAE:      {fmt_metric(overall_pred['mae'])}")
    print(f"    AUC:      {fmt_metric(overall_pred['auc'])}")

    # Per-source metrics
    print(f"\n  Per source ({len(per_source_pred)} sources):")
    print(f"    {'Source':<40} {'N':>6} {'BCE':>8} {'Acc':>8} {'MAE':>8} {'AUC':>8}")
    print(f"    {'-' * 40} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for source in sorted(per_source_pred):
        items = per_source_pred[source]
        m = compute_metrics([p for p, _, _ in items], [t for _, t, _ in items])
        print(f"    {source[:40]:<40} {m['n']:>6} {fmt_metric(m['bce']):>8} {fmt_metric(m['accuracy']):>8} {fmt_metric(m['mae']):>8} {fmt_metric(m['auc']):>8}")

    # Per-position accuracy (binned)
    print(f"\n  Accuracy by history length:")
    max_pos = max(per_position_pred.keys()) if per_position_pred else 0
    bin_size = max(1, max(10, (max_pos + 1) // 15))
    pos_bins = defaultdict(list)
    for pos, items in per_position_pred.items():
        b = (pos // bin_size) * bin_size
        pos_bins[b].extend(items)
    print(f"    {'Positions':<15} {'N':>6} {'Acc':>8} {'BCE':>8}")
    print(f"    {'-' * 15} {'-' * 6} {'-' * 8} {'-' * 8}")
    for b in sorted(pos_bins):
        items = pos_bins[b]
        m = compute_metrics([p for p, _ in items], [t for _, t in items])
        print(f"    {f'{b}-{b + bin_size - 1}':<15} {m['n']:>6} {fmt_metric(m['accuracy']):>8} {fmt_metric(m['bce']):>8}")

    # Calibration
    print(f"\n  Calibration (10 bins):")
    cal = calibration_bins(preds_all, targets_all)
    print(f"    {'Bin':<12} {'Mean Pred':>10} {'Mean Target':>12} {'Count':>8}")
    print(f"    {'-' * 12} {'-' * 10} {'-' * 12} {'-' * 8}")
    for c in cal:
        print(f"    {c['bin']:<12} {c['mean_pred']:>10.4f} {c['mean_target']:>12.4f} {c['count']:>8}")

    # ===================================================================
    # 2. Autoregressive forecasting (sweep over prefix fractions)
    # ===================================================================
    print("\n" + "=" * 70)
    print("AUTOREGRESSIVE FORECAST (prefix = 10%, 20%, ..., 90%)")
    print("=" * 70)

    # Build index: for each learner+source, store the prediction results by position
    # so we can efficiently extract the "predict" baseline on the forecasted portion
    pred_by_learner = []  # list parallel to learners_data, each is {pos: (pred, target)}
    for i, learner in enumerate(learners_data):
        source = learner["source"]
        lookup = {}
        for pred, target, pos, src in all_pred_results:
            if src == source:
                lookup[pos] = (pred, target)
        pred_by_learner.append(lookup)

    # Per-split results
    forecast_summary = []  # list of (frac, overall_metrics, time)
    all_fc_by_split = {}  # frac -> [(pred, target, horizon, source)]

    for frac in FORECAST_SPLITS:
        t0 = time.time()
        fc_results = []
        for i, learner in enumerate(learners_data):
            lid = f"bench_fc_{frac:.0%}_{i}"
            tasks = learner["tasks"]
            outcomes = learner["outcomes"]
            split_idx = max(1, int(len(tasks) * frac))
            if split_idx >= len(tasks):
                continue
            results = evaluate_forecast(client, lid, tasks, outcomes, split_idx)
            for pred, target, horizon in results:
                fc_results.append((pred, target, horizon, learner["source"]))
            print(f"\r  prefix={frac:.0%} [{i + 1}/{len(learners_data)}]", end="", flush=True)
        fc_time = time.time() - t0

        all_fc_by_split[frac] = fc_results
        if fc_results:
            m = compute_metrics([r[0] for r in fc_results], [r[1] for r in fc_results])
        else:
            m = compute_metrics([], [])
        forecast_summary.append((frac, m, fc_time))
        print(f"\r  prefix={frac:.0%}: {m['n']:>6} steps, acc={fmt_metric(m['accuracy'])}, bce={fmt_metric(m['bce'])} ({fc_time:.1f}s)")

    # Summary table
    print(f"\n  {'Prefix':>8} {'N':>8} {'BCE':>8} {'Acc':>8} {'MAE':>8} {'AUC':>8} {'Time':>7}")
    print(f"  {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 7}")
    for frac, m, t in forecast_summary:
        print(f"  {frac:>7.0%} {m['n']:>8} {fmt_metric(m['bce']):>8} {fmt_metric(m['accuracy']):>8} {fmt_metric(m['mae']):>8} {fmt_metric(m['auc']):>8} {t:>6.1f}s")

    # ===================================================================
    # 3. Forecast accuracy by horizon (per split)
    # ===================================================================
    print("\n" + "=" * 70)
    print("FORECAST ACCURACY BY HORIZON")
    print("=" * 70)

    for frac in FORECAST_SPLITS:
        fc_results = all_fc_by_split[frac]
        if not fc_results:
            continue
        per_horizon = defaultdict(list)
        for pred, target, horizon, _ in fc_results:
            per_horizon[horizon].append((pred, target))
        max_h = max(per_horizon.keys())
        h_bin_size = max(1, max(5, (max_h + 1) // 10))
        h_bins = defaultdict(list)
        for h, items in per_horizon.items():
            b = (h // h_bin_size) * h_bin_size
            h_bins[b].extend(items)

        print(f"\n  prefix={frac:.0%}:")
        print(f"    {'Horizon':<15} {'N':>6} {'Acc':>8} {'BCE':>8}")
        print(f"    {'-' * 15} {'-' * 6} {'-' * 8} {'-' * 8}")
        for b in sorted(h_bins):
            items = h_bins[b]
            m = compute_metrics([p for p, _ in items], [t for _, t in items])
            print(f"    {f'{b}-{b + h_bin_size - 1}':<15} {m['n']:>6} {fmt_metric(m['accuracy']):>8} {fmt_metric(m['bce']):>8}")

    # ===================================================================
    # 4. Prediction vs Forecast comparison
    # ===================================================================
    print("\n" + "=" * 70)
    print("PREDICTION vs FORECAST (same positions, true vs predicted outcomes)")
    print("=" * 70)

    print(f"\n  {'Prefix':>8} {'':>5} {'BCE':>8} {'Acc':>8} {'MAE':>8} {'AUC':>8}")
    print(f"  {'-' * 8} {'-' * 5} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    for frac in FORECAST_SPLITS:
        fc_results = all_fc_by_split[frac]
        if not fc_results:
            continue

        # Collect the prediction-mode results for the same positions
        pred_on_fc = []
        fc_on_fc = []
        for i, learner in enumerate(learners_data):
            split_idx = max(1, int(len(learner["tasks"]) * frac))
            if split_idx >= len(learner["tasks"]):
                continue
            lookup = pred_by_learner[i]
            source = learner["source"]
            src_fc = [r for r in fc_results if r[3] == source]
            for pred_fc, target_fc, horizon, _ in src_fc:
                pos = split_idx + horizon
                if pos in lookup:
                    p_pred, t_pred = lookup[pos]
                    pred_on_fc.append((p_pred, t_pred))
                    fc_on_fc.append((pred_fc, target_fc))

        if not pred_on_fc:
            continue

        m_pred = compute_metrics([p for p, _ in pred_on_fc], [t for _, t in pred_on_fc])
        m_fc = compute_metrics([p for p, _ in fc_on_fc], [t for _, t in fc_on_fc])
        print(f"  {frac:>7.0%} {'pred':>5} {fmt_metric(m_pred['bce']):>8} {fmt_metric(m_pred['accuracy']):>8} {fmt_metric(m_pred['mae']):>8} {fmt_metric(m_pred['auc']):>8}")
        print(f"  {'':>7} {'fc':>5} {fmt_metric(m_fc['bce']):>8} {fmt_metric(m_fc['accuracy']):>8} {fmt_metric(m_fc['mae']):>8} {fmt_metric(m_fc['auc']):>8}")
        # Delta row
        deltas = {}
        for k in ["bce", "accuracy", "mae"]:
            if m_pred[k] is not None and m_fc[k] is not None:
                deltas[k] = f"{m_fc[k] - m_pred[k]:>+8.4f}"
            else:
                deltas[k] = "     N/A"
        if m_pred["auc"] is not None and m_fc["auc"] is not None:
            deltas["auc"] = f"{m_fc['auc'] - m_pred['auc']:>+8.4f}"
        else:
            deltas["auc"] = "     N/A"
        print(f"  {'':>7} {'delta':>5} {deltas['bce']:>8} {deltas['accuracy']:>8} {deltas['mae']:>8} {deltas['auc']:>8}")

    # ===================================================================
    # 5. Plots
    # ===================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        plot_dir = args.data.replace(".jsonl", "") + "_plots"
        import os
        os.makedirs(plot_dir, exist_ok=True)

        fracs = [frac for frac, m, _ in forecast_summary if m["accuracy"] is not None]
        fc_accs = [m["accuracy"] for _, m, _ in forecast_summary if m["accuracy"] is not None]
        fc_bces = [m["bce"] for _, m, _ in forecast_summary if m["bce"] is not None]
        fc_aucs = [m["auc"] for _, m, _ in forecast_summary if m["auc"] is not None]
        fc_maes = [m["mae"] for _, m, _ in forecast_summary if m["mae"] is not None]

        # Prediction-mode baselines on the forecasted portions
        pred_accs, pred_bces, pred_aucs, pred_maes = [], [], [], []
        for frac in fracs:
            split_preds, split_targets = [], []
            for i, learner in enumerate(learners_data):
                split_idx = max(1, int(len(learner["tasks"]) * frac))
                if split_idx >= len(learner["tasks"]):
                    continue
                lookup = pred_by_learner[i]
                for pos in range(split_idx, len(learner["tasks"])):
                    if pos in lookup:
                        p, t = lookup[pos]
                        split_preds.append(p)
                        split_targets.append(t)
            m = compute_metrics(split_preds, split_targets)
            pred_accs.append(m["accuracy"])
            pred_bces.append(m["bce"])
            pred_aucs.append(m["auc"])
            pred_maes.append(m["mae"])

        # --- Fig 1: Forecast vs prefix fraction (4 metrics) ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, fc_vals, pred_vals, ylabel, title in [
            (axes[0, 0], fc_accs, pred_accs, "Accuracy", "Accuracy vs Prefix"),
            (axes[0, 1], fc_bces, pred_bces, "BCE", "BCE vs Prefix"),
            (axes[1, 0], fc_maes, pred_maes, "MAE", "MAE vs Prefix"),
            (axes[1, 1], fc_aucs, pred_aucs, "AUC", "AUC vs Prefix"),
        ]:
            ax.plot(fracs, fc_vals, "o-", label="Forecast", color="tab:blue")
            ax.plot(fracs, pred_vals, "s--", label="Prediction", color="tab:orange")
            ax.set_xlabel("Prefix fraction")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Forecast vs Prediction by Prefix Fraction", fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/forecast_vs_prefix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- Fig 2: Calibration reliability diagram (prediction) ---
        cal = calibration_bins(preds_all, targets_all, n_bins=10)
        if cal:
            fig, ax = plt.subplots(figsize=(6, 6))
            mean_preds = [c["mean_pred"] for c in cal]
            mean_targets = [c["mean_target"] for c in cal]
            counts = [c["count"] for c in cal]
            ax.bar(mean_preds, mean_targets, width=0.08, alpha=0.6, label="Observed", color="tab:blue", edgecolor="white")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Observed frequency")
            ax.set_title("Calibration — Next-step Prediction")
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            # Add count annotations
            for mp, mt, c in zip(mean_preds, mean_targets, counts):
                ax.annotate(f"n={c}", (mp, mt), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7, color="gray")
            fig.tight_layout()
            fig.savefig(f"{plot_dir}/calibration.png", dpi=150)
            plt.close(fig)

        # --- Fig 3: Prediction distribution by outcome class ---
        fig, ax = plt.subplots(figsize=(8, 5))
        preds_pos = [p for p, t in zip(preds_all, targets_all) if t >= 0.5]
        preds_neg = [p for p, t in zip(preds_all, targets_all) if t < 0.5]
        bins = np.linspace(0, 1, 30)
        ax.hist(preds_neg, bins=bins, alpha=0.6, label=f"Outcome=0 (n={len(preds_neg)})", color="tab:red", density=True)
        ax.hist(preds_pos, bins=bins, alpha=0.6, label=f"Outcome=1 (n={len(preds_pos)})", color="tab:green", density=True)
        ax.axvline(0.5, color="black", linestyle="--", alpha=0.4)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        ax.set_title("Prediction Distribution by True Outcome")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/prediction_distribution.png", dpi=150)
        plt.close(fig)

        # --- Fig 4: Accuracy by history length ---
        if per_position_pred:
            fig, ax = plt.subplots(figsize=(10, 5))
            sorted_bins = sorted(pos_bins.keys())
            bin_accs = []
            bin_ns = []
            bin_labels = []
            for b in sorted_bins:
                items = pos_bins[b]
                m = compute_metrics([p for p, _ in items], [t for _, t in items])
                bin_accs.append(m["accuracy"])
                bin_ns.append(m["n"])
                bin_labels.append(f"{b}-{b + bin_size - 1}")
            ax.bar(range(len(sorted_bins)), bin_accs, color="tab:blue", alpha=0.7, edgecolor="white")
            ax.set_xticks(range(len(sorted_bins)))
            ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("History length (position)")
            ax.set_ylabel("Accuracy")
            ax.set_title("Prediction Accuracy by History Length")
            ax.grid(True, alpha=0.3, axis="y")
            # Add count annotations
            for i, (acc, n) in enumerate(zip(bin_accs, bin_ns)):
                if acc is not None:
                    ax.annotate(f"n={n}", (i, acc), textcoords="offset points",
                                xytext=(0, 5), ha="center", fontsize=7, color="gray")
            fig.tight_layout()
            fig.savefig(f"{plot_dir}/accuracy_by_history.png", dpi=150)
            plt.close(fig)

        # --- Fig 5: Per-source accuracy bar chart ---
        if per_source_pred:
            sources_sorted = sorted(per_source_pred.keys())
            src_accs = []
            src_ns = []
            for source in sources_sorted:
                items = per_source_pred[source]
                m = compute_metrics([p for p, _, _ in items], [t for _, t, _ in items])
                src_accs.append(m["accuracy"])
                src_ns.append(m["n"])
            fig, ax = plt.subplots(figsize=(max(8, len(sources_sorted) * 0.5), 6))
            bars = ax.barh(range(len(sources_sorted)), src_accs, color="tab:blue", alpha=0.7, edgecolor="white")
            ax.set_yticks(range(len(sources_sorted)))
            labels = [s[:40] for s in sources_sorted]
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("Accuracy")
            ax.set_title("Prediction Accuracy by Source")
            ax.axvline(overall_pred["accuracy"], color="red", linestyle="--", alpha=0.6, label=f"Overall: {overall_pred['accuracy']:.3f}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="x")
            # Count annotations
            for i, (acc, n) in enumerate(zip(src_accs, src_ns)):
                if acc is not None:
                    ax.annotate(f" n={n}", (acc, i), va="center", fontsize=7, color="gray")
            fig.tight_layout()
            fig.savefig(f"{plot_dir}/accuracy_by_source.png", dpi=150)
            plt.close(fig)

        # --- Fig 6: Forecast accuracy by horizon (multiple splits overlaid) ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        cmap = plt.cm.viridis
        for idx, frac in enumerate(FORECAST_SPLITS):
            fc_results = all_fc_by_split.get(frac, [])
            if not fc_results:
                continue
            per_horizon = defaultdict(list)
            for pred, target, horizon, _ in fc_results:
                per_horizon[horizon].append((pred, target))
            max_h = max(per_horizon.keys()) if per_horizon else 0
            h_bin = max(1, max(5, (max_h + 1) // 15))
            h_bins_fc = defaultdict(list)
            for h, items in per_horizon.items():
                b = (h // h_bin) * h_bin
                h_bins_fc[b].extend(items)

            horizons = sorted(h_bins_fc.keys())
            accs = []
            bces = []
            for b in horizons:
                items = h_bins_fc[b]
                m = compute_metrics([p for p, _ in items], [t for _, t in items])
                accs.append(m["accuracy"])
                bces.append(m["bce"])

            color = cmap(idx / max(1, len(FORECAST_SPLITS) - 1))
            ax1.plot(horizons, accs, "o-", color=color, label=f"{frac:.0%}", markersize=3, alpha=0.8)
            ax2.plot(horizons, bces, "o-", color=color, label=f"{frac:.0%}", markersize=3, alpha=0.8)

        ax1.set_xlabel("Forecast horizon (steps ahead)")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Forecast Accuracy by Horizon")
        ax1.legend(title="Prefix", fontsize=7, title_fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Forecast horizon (steps ahead)")
        ax2.set_ylabel("BCE")
        ax2.set_title("Forecast BCE by Horizon")
        ax2.legend(title="Prefix", fontsize=7, title_fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Forecast Quality by Horizon", fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/forecast_by_horizon.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- Fig 7: Ground truth vs forecast scatter (50% split as representative) ---
        representative_frac = 0.5
        fc_results_rep = all_fc_by_split.get(representative_frac, [])
        if fc_results_rep:
            fc_preds_rep = [r[0] for r in fc_results_rep]
            fc_targets_rep = [r[1] for r in fc_results_rep]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Prediction: predicted vs ground truth (2D histogram)
            ax1.hist2d(preds_all, targets_all, bins=[np.linspace(0, 1, 50), [-0.1, 0.5, 1.1]],
                       cmap="Blues", cmin=1)
            ax1.set_xlabel("Predicted probability")
            ax1.set_ylabel("Ground truth")
            ax1.set_title("Prediction: Predicted vs Ground Truth")
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(["0 (fail)", "1 (pass)"])

            # Forecast: predicted vs ground truth (2D histogram)
            ax2.hist2d(fc_preds_rep, fc_targets_rep, bins=[np.linspace(0, 1, 50), [-0.1, 0.5, 1.1]],
                       cmap="Oranges", cmin=1)
            ax2.set_xlabel("Predicted probability")
            ax2.set_ylabel("Ground truth")
            ax2.set_title(f"Forecast ({representative_frac:.0%} prefix): Predicted vs Ground Truth")
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(["0 (fail)", "1 (pass)"])

            fig.tight_layout()
            fig.savefig(f"{plot_dir}/predicted_vs_gt.png", dpi=150)
            plt.close(fig)

        # --- Fig 8: ROC curve (prediction) ---
        if overall_pred["auc"] is not None:
            fig, ax = plt.subplots(figsize=(6, 6))
            # Compute ROC points
            pairs = sorted(zip(preds_all, targets_all), key=lambda x: -x[0])
            n_pos = sum(1 for _, t in pairs if t >= 0.5)
            n_neg = len(pairs) - n_pos
            tpr_list, fpr_list = [0.0], [0.0]
            tp, fp = 0, 0
            prev_score = None
            for score, label in pairs:
                if score != prev_score and prev_score is not None:
                    fpr_list.append(fp / n_neg)
                    tpr_list.append(tp / n_pos)
                if label >= 0.5:
                    tp += 1
                else:
                    fp += 1
                prev_score = score
            fpr_list.append(fp / n_neg)
            tpr_list.append(tp / n_pos)

            ax.plot(fpr_list, tpr_list, color="tab:blue", linewidth=2,
                    label=f"Prediction (AUC={overall_pred['auc']:.3f})")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve — Next-step Prediction")
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{plot_dir}/roc_curve.png", dpi=150)
            plt.close(fig)

        print(f"\n  Plots saved to {plot_dir}/")
    except ImportError:
        print("\n  (matplotlib not installed — skipping plots)")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Learners:         {len(learners_data)}")
    print(f"  Total steps:      {total_steps:,}")
    print(f"  Prediction time:  {pred_time:.1f}s")
    total_fc_time = sum(t for _, _, t in forecast_summary)
    print(f"  Forecast time:    {total_fc_time:.1f}s (9 splits)")
    if overall_pred["accuracy"] is not None:
        print(f"  Prediction acc:   {overall_pred['accuracy']:.4f}")
    # Best forecast accuracy across splits
    best_fc = max(forecast_summary, key=lambda x: x[1]["accuracy"] or 0)
    if best_fc[1]["accuracy"] is not None:
        print(f"  Best forecast:    {best_fc[1]['accuracy']:.4f} (prefix={best_fc[0]:.0%})")


if __name__ == "__main__":
    main()
