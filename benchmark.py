"""
Benchmark a DFM model checkpoint via the inference server.

Usage:
    # Start the server first:
    DFM_CHECKPOINT_PATH=ckpt.pt DFM_EMBEDDINGS_PATH=embeddings.pt python -m dfm_server.server

    # Then run the benchmark:
    python benchmark.py --data val.jsonl [--server-url http://localhost:8000]
                        [--max-learners 0] [--output-json results.json]
"""

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import datetime

from dfm_server.client import DFMClient

FORECAST_SPLITS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

_KNOWN_SOURCES = [
    "dbe_kt22", "duolingo", "adaptivmath", "codeworkout",
    "xes3g5m", "eedi", "ollv2", "ollv1", "icl", "rl",
]


def _extract_source(name):
    """Extract dataset-level source label from trajectory name."""
    for src in _KNOWN_SOURCES:
        if name.startswith(src):
            return src
    return name.split("_")[0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(predictions, targets, include_correlation=False):
    """Compute BCE, accuracy, MAE, AUC, and optionally Pearson/Spearman."""
    n = len(predictions)
    if n == 0:
        out = {"bce": None, "accuracy": None, "mae": None, "auc": None, "n": 0}
        if include_correlation:
            out["pearson"] = None
            out["spearman"] = None
        return out

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

    out = {"bce": bce, "accuracy": accuracy, "mae": mae, "auc": auc, "n": n}
    if include_correlation:
        out["pearson"], out["spearman"] = _correlations(predictions, targets)
    return out


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


def _correlations(predictions, targets):
    """Compute Pearson and Spearman correlation. Returns (pearson, spearman)."""
    n = len(predictions)
    if n < 3:
        return None, None

    # Pearson
    mean_p = sum(predictions) / n
    mean_t = sum(targets) / n
    cov = sum((p - mean_p) * (t - mean_t) for p, t in zip(predictions, targets))
    var_p = sum((p - mean_p) ** 2 for p in predictions)
    var_t = sum((t - mean_t) ** 2 for t in targets)
    denom = (var_p * var_t) ** 0.5
    pearson = cov / denom if denom > 0 else None

    # Spearman (rank correlation = Pearson on ranks)
    def _rank(vals):
        indexed = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2.0
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    rank_p = _rank(predictions)
    rank_t = _rank(targets)
    mean_rp = sum(rank_p) / n
    mean_rt = sum(rank_t) / n
    cov_r = sum((rp - mean_rp) * (rt - mean_rt) for rp, rt in zip(rank_p, rank_t))
    var_rp = sum((rp - mean_rp) ** 2 for rp in rank_p)
    var_rt = sum((rt - mean_rt) ** 2 for rt in rank_t)
    denom_r = (var_rp * var_rt) ** 0.5
    spearman = cov_r / denom_r if denom_r > 0 else None

    return pearson, spearman


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
# Baselines
# ---------------------------------------------------------------------------


def compute_baselines(learners_data):
    """Compute baseline predictions (no server needed).

    Returns: {name: list of per-learner result lists}
    Each per-learner list contains (pred, target, pos) tuples.
    """
    all_outcomes = [o for l in learners_data for o in l["outcomes"]]
    global_mean = sum(all_outcomes) / len(all_outcomes) if all_outcomes else 0.5

    baselines = {"global_mean": [], "running_avg": [], "last_outcome": []}

    for learner in learners_data:
        outcomes = learner["outcomes"]
        gm_results = []
        ra_results = []
        lo_results = []
        running_sum = 0.0
        for i, outcome in enumerate(outcomes):
            gm_results.append((global_mean, outcome, i))
            ra_results.append((running_sum / i if i > 0 else global_mean, outcome, i))
            lo_results.append((outcomes[i - 1] if i > 0 else global_mean, outcome, i))
            running_sum += outcome
        baselines["global_mean"].append(gm_results)
        baselines["running_avg"].append(ra_results)
        baselines["last_outcome"].append(lo_results)

    return baselines


def compute_per_learner_metrics(per_learner_results, include_correlation=False, min_steps=5):
    """Compute metrics per learner, return mean and std across learners.

    Args:
        per_learner_results: list of [(pred, target, pos), ...] per learner
        include_correlation: whether to include Pearson/Spearman
        min_steps: skip learners with fewer steps

    Returns: {"mean": {...}, "std": {...}, "n_learners": int}
    """
    keys = ["bce", "accuracy", "mae", "auc"]
    if include_correlation:
        keys += ["pearson", "spearman"]

    learner_metrics = []
    for results in per_learner_results:
        if len(results) < min_steps:
            continue
        m = compute_metrics(
            [r[0] for r in results], [r[1] for r in results],
            include_correlation=include_correlation,
        )
        learner_metrics.append(m)

    if not learner_metrics:
        return {"mean": {k: None for k in keys}, "std": {k: None for k in keys}, "n_learners": 0}

    mean_metrics = {}
    std_metrics = {}
    for key in keys:
        values = [m[key] for m in learner_metrics if m[key] is not None]
        if values:
            mean_metrics[key] = sum(values) / len(values)
            var = sum((v - mean_metrics[key]) ** 2 for v in values) / len(values)
            std_metrics[key] = var ** 0.5
        else:
            mean_metrics[key] = None
            std_metrics[key] = None

    return {"mean": mean_metrics, "std": std_metrics, "n_learners": len(learner_metrics)}


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------


def evaluate_prediction(client, learner_id, tasks, outcomes, max_steps=None):
    """Next-step prediction: register empty, then predict+update for each step.

    Args:
        max_steps: If set, truncate to this many steps to avoid KV cache overflow.

    Returns list of (prediction, target, position) tuples.
    """
    if max_steps and len(tasks) > max_steps:
        tasks = tasks[:max_steps]
        outcomes = outcomes[:max_steps]
    client.register(learner_id)
    results = []
    for i, (task, outcome) in enumerate(zip(tasks, outcomes)):
        preds = client.predict(learner_id, [task])
        results.append((preds[0], outcome, i))
        client.update(learner_id, task, outcome)
    client.delete(learner_id)
    return results


def evaluate_forecast(client, learner_id, tasks, outcomes, split_idx, max_steps=None):
    """Autoregressive forecast: register with history[:split], forecast the rest.

    Args:
        max_steps: If set, truncate total length to avoid KV cache overflow.

    Returns list of (prediction, target, horizon) tuples.
    """
    if max_steps and len(tasks) > max_steps:
        tasks = tasks[:max_steps]
        outcomes = outcomes[:max_steps]
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
                        help="Max learners per source to evaluate (0 = all)")
    parser.add_argument("--output-json", default=None,
                        help="Path to write JSON results (optional)")
    args = parser.parse_args()

    # Check server
    client = DFMClient(args.server_url)
    health = client.health()
    print(f"Server OK: {health['learner_count']} learners registered")
    model_config = client.config()
    # KV cache limit: 2 interleaved tokens per step + 2 for BOS
    max_steps = model_config.get("block_size", 4096) // 2 - 1
    print(f"Max steps per learner: {max_steps}")

    # Load data
    learners_data = []
    with open(args.data) as f:
        for line in f:
            learners_data.append(json.loads(line))
    print(f"Loaded {len(learners_data)} learners from {args.data}")

    # Assign dataset-level source to each learner
    for learner in learners_data:
        learner["dataset_source"] = _extract_source(learner["source"])

    # Cap to max_learners per source
    if args.max_learners > 0:
        by_source = defaultdict(list)
        for learner in learners_data:
            by_source[learner["dataset_source"]].append(learner)
        learners_data = []
        for source in sorted(by_source):
            learners_data.extend(by_source[source][: args.max_learners])
        print(f"Capped to {args.max_learners} learners per source: {len(learners_data)} total")

    total_steps = sum(len(l["tasks"]) for l in learners_data)
    print(f"Total steps: {total_steps:,}")

    # Detect graded outcomes
    has_graded = any(
        o not in (0.0, 1.0) for l in learners_data for o in l["outcomes"]
    )

    # JSON output accumulator
    json_out = {
        "metadata": {
            "data_path": args.data,
            "server_url": args.server_url,
            "timestamp": datetime.now().isoformat(),
            "n_learners": len(learners_data),
            "total_steps": total_steps,
            "model_config": model_config,
            "has_graded_outcomes": has_graded,
        },
    }

    # ===================================================================
    # 0. Baselines (no server needed)
    # ===================================================================
    print("\n" + "=" * 70)
    print("BASELINES (computed from val data, no model)")
    print("=" * 70)

    baseline_per_learner = compute_baselines(learners_data)
    baseline_overall = {}
    json_out["baselines"] = {}

    for bl_name, per_learner_results in baseline_per_learner.items():
        all_results = [r for results in per_learner_results for r in results]
        preds = [r[0] for r in all_results]
        tgts = [r[1] for r in all_results]
        overall = compute_metrics(preds, tgts, include_correlation=has_graded)
        baseline_overall[bl_name] = overall
        pl_stats = compute_per_learner_metrics(per_learner_results, include_correlation=has_graded)

        # Per-source
        per_source = defaultdict(list)
        for learner, results in zip(learners_data, per_learner_results):
            per_source[learner["dataset_source"]].extend(results)
        per_source_metrics = {}
        for src, items in sorted(per_source.items()):
            per_source_metrics[src] = compute_metrics(
                [r[0] for r in items], [r[1] for r in items], include_correlation=has_graded,
            )

        json_out["baselines"][bl_name] = {
            "overall": overall,
            "per_source": per_source_metrics,
            "per_learner": pl_stats,
        }

    print(f"\n  {'Method':<16} {'N':>6} {'BCE':>8} {'Acc':>8} {'MAE':>8} {'AUC':>8}", end="")
    if has_graded:
        print(f" {'Pearson':>8} {'Spearman':>8}", end="")
    print()
    print(f"  {'-' * 16} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}", end="")
    if has_graded:
        print(f" {'-' * 8} {'-' * 8}", end="")
    print()
    for bl_name, m in baseline_overall.items():
        print(f"  {bl_name:<16} {m['n']:>6} {fmt_metric(m['bce']):>8} {fmt_metric(m['accuracy']):>8} {fmt_metric(m['mae']):>8} {fmt_metric(m['auc']):>8}", end="")
        if has_graded:
            print(f" {fmt_metric(m.get('pearson')):>8} {fmt_metric(m.get('spearman')):>8}", end="")
        print()

    # ===================================================================
    # 1. Next-step prediction
    # ===================================================================
    print("\n" + "=" * 70)
    print("NEXT-STEP PREDICTION")
    print("=" * 70)

    per_learner_pred = []  # list of per-learner result lists
    per_source_pred = defaultdict(list)  # dataset_source -> [(pred, target, pos)]
    per_position_pred = defaultdict(list)

    t0 = time.time()
    for i, learner in enumerate(learners_data):
        lid = f"bench_pred_{i}"
        tasks = learner["tasks"]
        outcomes = learner["outcomes"]
        ds = learner["dataset_source"]
        results = evaluate_prediction(client, lid, tasks, outcomes, max_steps)
        per_learner_pred.append(results)
        for pred, target, pos in results:
            per_source_pred[ds].append((pred, target, pos))
            per_position_pred[pos].append((pred, target))
        print(f"\r  [{i + 1}/{len(learners_data)}] {ds} ({len(tasks)} steps)", end="", flush=True)
    pred_time = time.time() - t0
    print(f"\n  Completed in {pred_time:.1f}s ({total_steps / pred_time:.0f} steps/s)")

    # Overall metrics
    all_pred_flat = [r for results in per_learner_pred for r in results]
    preds_all = [r[0] for r in all_pred_flat]
    targets_all = [r[1] for r in all_pred_flat]
    overall_pred = compute_metrics(preds_all, targets_all, include_correlation=has_graded)

    print(f"\n  Overall ({overall_pred['n']:,} predictions):")
    print(f"    BCE:      {fmt_metric(overall_pred['bce'])}")
    print(f"    Accuracy: {fmt_metric(overall_pred['accuracy'])}")
    print(f"    MAE:      {fmt_metric(overall_pred['mae'])}")
    print(f"    AUC:      {fmt_metric(overall_pred['auc'])}")
    if has_graded:
        print(f"    Pearson:  {fmt_metric(overall_pred.get('pearson'))}")
        print(f"    Spearman: {fmt_metric(overall_pred.get('spearman'))}")

    # Per-learner statistics
    pl_stats = compute_per_learner_metrics(per_learner_pred, include_correlation=has_graded)
    print(f"\n  Per-learner (N={pl_stats['n_learners']}):", end="")
    for key in ["bce", "accuracy", "mae", "auc"]:
        m = pl_stats["mean"].get(key)
        s = pl_stats["std"].get(key)
        if m is not None and s is not None:
            print(f"  {key.upper()} {m:.4f}+-{s:.4f}", end="")
    print()

    # Per-source metrics
    per_source_metrics = {}
    print(f"\n  Per source ({len(per_source_pred)} sources):")
    print(f"    {'Source':<20} {'N':>6} {'BCE':>8} {'Acc':>8} {'MAE':>8} {'AUC':>8}")
    print(f"    {'-' * 20} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for source in sorted(per_source_pred):
        items = per_source_pred[source]
        m = compute_metrics([p for p, _, _ in items], [t for _, t, _ in items], include_correlation=has_graded)
        per_source_metrics[source] = m
        print(f"    {source:<20} {m['n']:>6} {fmt_metric(m['bce']):>8} {fmt_metric(m['accuracy']):>8} {fmt_metric(m['mae']):>8} {fmt_metric(m['auc']):>8}")

    # Model vs baseline comparison
    print(f"\n  Model vs Baselines:")
    print(f"    {'Method':<16} {'BCE':>8} {'Acc':>8} {'MAE':>8} {'AUC':>8}")
    print(f"    {'-' * 16} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    print(f"    {'DFM':<16} {fmt_metric(overall_pred['bce']):>8} {fmt_metric(overall_pred['accuracy']):>8} {fmt_metric(overall_pred['mae']):>8} {fmt_metric(overall_pred['auc']):>8}")
    for bl_name, m in baseline_overall.items():
        print(f"    {bl_name:<16} {fmt_metric(m['bce']):>8} {fmt_metric(m['accuracy']):>8} {fmt_metric(m['mae']):>8} {fmt_metric(m['auc']):>8}")

    json_out["prediction"] = {
        "overall": overall_pred,
        "per_source": per_source_metrics,
        "per_learner": pl_stats,
    }
    json_out["metadata"]["prediction_time_s"] = pred_time

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

    # Build per-learner prediction lookup for forecast comparison
    pred_by_learner = []
    for i, results in enumerate(per_learner_pred):
        lookup = {pos: (pred, target) for pred, target, pos in results}
        pred_by_learner.append(lookup)

    # Per-split results
    forecast_summary = []  # list of (frac, overall_metrics, time)
    all_fc_by_split = {}  # frac -> [(pred, target, horizon, dataset_source)]

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
            results = evaluate_forecast(client, lid, tasks, outcomes, split_idx, max_steps)
            for pred, target, horizon in results:
                fc_results.append((pred, target, horizon, learner["dataset_source"]))
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

    total_fc_time = sum(t for _, _, t in forecast_summary)

    json_out["forecast"] = {
        f"{frac}": m for frac, m, _ in forecast_summary
    }
    json_out["metadata"]["forecast_time_s"] = total_fc_time

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

        # Group forecast results by learner index
        fc_by_learner = defaultdict(list)
        learner_idx = 0
        for i, learner in enumerate(learners_data):
            split_idx = max(1, int(len(learner["tasks"]) * frac))
            if split_idx >= len(learner["tasks"]):
                continue
            fc_by_learner[i] = []

        fc_offset = 0
        for i, learner in enumerate(learners_data):
            split_idx = max(1, int(len(learner["tasks"]) * frac))
            if split_idx >= len(learner["tasks"]):
                continue
            n_future = len(learner["tasks"]) - split_idx
            for h in range(n_future):
                if fc_offset < len(fc_results):
                    fc_by_learner[i].append(fc_results[fc_offset])
                    fc_offset += 1

        pred_on_fc = []
        fc_on_fc = []
        for i in fc_by_learner:
            split_idx = max(1, int(len(learners_data[i]["tasks"]) * frac))
            lookup = pred_by_learner[i]
            for pred_fc, target_fc, horizon, _ in fc_by_learner[i]:
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
            ax.barh(range(len(sources_sorted)), src_accs, color="tab:blue", alpha=0.7, edgecolor="white")
            ax.set_yticks(range(len(sources_sorted)))
            ax.set_yticklabels(sources_sorted, fontsize=9)
            ax.set_xlabel("Accuracy")
            ax.set_title("Prediction Accuracy by Source")
            ax.axvline(overall_pred["accuracy"], color="red", linestyle="--", alpha=0.6, label=f"Overall: {overall_pred['accuracy']:.3f}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="x")
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

            ax1.hist2d(preds_all, targets_all, bins=[np.linspace(0, 1, 50), [-0.1, 0.5, 1.1]],
                       cmap="Blues", cmin=1)
            ax1.set_xlabel("Predicted probability")
            ax1.set_ylabel("Ground truth")
            ax1.set_title("Prediction: Predicted vs Ground Truth")
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(["0 (fail)", "1 (pass)"])

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

        # --- Fig 9: Model vs Baselines comparison ---
        methods = ["DFM"] + list(baseline_overall.keys())
        method_metrics = [overall_pred] + [baseline_overall[bl] for bl in baseline_overall]
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        colors = ["tab:blue"] + ["tab:gray"] * len(baseline_overall)
        for ax, metric, title in zip(axes, ["accuracy", "bce", "mae", "auc"],
                                     ["Accuracy", "BCE", "MAE", "AUC"]):
            values = [m.get(metric) for m in method_metrics]
            valid = [(meth, v) for meth, v in zip(methods, values) if v is not None]
            if not valid:
                continue
            meths, vals = zip(*valid)
            c = colors[:len(meths)]
            ax.barh(range(len(meths)), vals, color=c, alpha=0.7, edgecolor="white")
            ax.set_yticks(range(len(meths)))
            ax.set_yticklabels(meths)
            ax.set_xlabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="x")
            for i, v in enumerate(vals):
                ax.annotate(f" {v:.3f}", (v, i), va="center", fontsize=8)
        fig.suptitle("Model vs Baselines", fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/model_vs_baselines.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- Fig 10: Per-source model vs running_avg ---
        if per_source_pred and "running_avg" in baseline_per_learner:
            bl_per_source = defaultdict(list)
            for learner, results in zip(learners_data, baseline_per_learner["running_avg"]):
                bl_per_source[learner["dataset_source"]].extend(results)

            sources_sorted = sorted(per_source_pred.keys())
            model_accs = []
            bl_accs = []
            for src in sources_sorted:
                m_model = compute_metrics(
                    [p for p, _, _ in per_source_pred[src]],
                    [t for _, t, _ in per_source_pred[src]],
                )
                model_accs.append(m_model["accuracy"])
                if src in bl_per_source:
                    m_bl = compute_metrics(
                        [r[0] for r in bl_per_source[src]],
                        [r[1] for r in bl_per_source[src]],
                    )
                    bl_accs.append(m_bl["accuracy"])
                else:
                    bl_accs.append(None)

            fig, ax = plt.subplots(figsize=(max(8, len(sources_sorted) * 0.8), 6))
            y = range(len(sources_sorted))
            ax.barh([i - 0.15 for i in y], model_accs, height=0.3, label="DFM", color="tab:blue", alpha=0.8)
            bl_accs_valid = [v if v is not None else 0 for v in bl_accs]
            ax.barh([i + 0.15 for i in y], bl_accs_valid, height=0.3, label="Running Avg", color="tab:orange", alpha=0.8)
            ax.set_yticks(list(y))
            ax.set_yticklabels(sources_sorted, fontsize=9)
            ax.set_xlabel("Accuracy")
            ax.set_title("DFM vs Running Average by Source")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="x")
            fig.tight_layout()
            fig.savefig(f"{plot_dir}/per_source_vs_baseline.png", dpi=150)
            plt.close(fig)

        print(f"\n  Plots saved to {plot_dir}/")
    except ImportError:
        print("\n  (matplotlib not installed -- skipping plots)")

    # ===================================================================
    # JSON output
    # ===================================================================
    if args.output_json:
        # Make all values JSON-serializable
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return round(obj, 6)
            if hasattr(obj, "item"):  # torch/numpy scalar
                return round(float(obj), 6)
            return obj

        with open(args.output_json, "w") as f:
            json.dump(_clean(json_out), f, indent=2)
        print(f"\n  JSON results saved to {args.output_json}")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Learners:         {len(learners_data)}")
    print(f"  Total steps:      {total_steps:,}")
    print(f"  Prediction time:  {pred_time:.1f}s")
    print(f"  Forecast time:    {total_fc_time:.1f}s (9 splits)")
    if overall_pred["accuracy"] is not None:
        print(f"  Prediction acc:   {overall_pred['accuracy']:.4f}")
        # Show lift over best baseline
        best_bl_acc = max(m["accuracy"] for m in baseline_overall.values() if m["accuracy"] is not None)
        delta = overall_pred["accuracy"] - best_bl_acc
        print(f"  vs best baseline: {delta:+.4f}")
    best_fc = max(forecast_summary, key=lambda x: x[1]["accuracy"] or 0)
    if best_fc[1]["accuracy"] is not None:
        print(f"  Best forecast:    {best_fc[1]['accuracy']:.4f} (prefix={best_fc[0]:.0%})")


if __name__ == "__main__":
    main()
