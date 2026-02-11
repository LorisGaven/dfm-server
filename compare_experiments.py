"""
Compare benchmark results across multiple generalization experiments.

Usage:
    python compare_experiments.py results/*.json
    python compare_experiments.py --sort accuracy results/*.json
    python compare_experiments.py --metric bce --output table.csv results/*.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_result(path):
    with open(path) as f:
        return json.load(f)


def fmt(v, f=".4f"):
    return f"{v:{f}}" if v is not None else "N/A"


def fmt_delta(v, f="+.4f"):
    return f"{v:{f}}" if v is not None else "N/A"


def main():
    parser = argparse.ArgumentParser(description="Compare DFM benchmark results")
    parser.add_argument("results", nargs="+", help="Paths to JSON result files from benchmark.py")
    parser.add_argument("--sort", default="accuracy", choices=["accuracy", "bce", "mae", "auc", "name"],
                        help="Sort experiments by this metric (default: accuracy)")
    parser.add_argument("--metric", default=None,
                        help="Show detailed per-source breakdown for this metric")
    parser.add_argument("--output", default=None,
                        help="Write comparison table as CSV")
    args = parser.parse_args()

    # Load all results
    experiments = []
    for path in args.results:
        try:
            data = load_result(path)
            name = Path(path).stem
            experiments.append({"name": name, "path": path, "data": data})
        except Exception as e:
            print(f"Warning: could not load {path}: {e}", file=sys.stderr)

    if not experiments:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(experiments)} experiments\n")

    # ===================================================================
    # 1. Overall comparison table
    # ===================================================================
    print("=" * 90)
    print("OVERALL COMPARISON")
    print("=" * 90)

    rows = []
    for exp in experiments:
        pred = exp["data"].get("prediction", {}).get("overall", {})
        meta = exp["data"].get("metadata", {})
        baselines = exp["data"].get("baselines", {})

        # Best baseline accuracy (running_avg is usually best)
        best_bl_acc = None
        best_bl_name = None
        for bl_name, bl_data in baselines.items():
            bl_acc = bl_data.get("overall", {}).get("accuracy")
            if bl_acc is not None and (best_bl_acc is None or bl_acc > best_bl_acc):
                best_bl_acc = bl_acc
                best_bl_name = bl_name

        lift = None
        if pred.get("accuracy") is not None and best_bl_acc is not None:
            lift = pred["accuracy"] - best_bl_acc

        # Val sources
        sources = sorted(exp["data"].get("prediction", {}).get("per_source", {}).keys())
        sources_str = ",".join(sources) if sources else "?"

        rows.append({
            "name": exp["name"],
            "sources": sources_str,
            "n": pred.get("n", 0),
            "bce": pred.get("bce"),
            "accuracy": pred.get("accuracy"),
            "mae": pred.get("mae"),
            "auc": pred.get("auc"),
            "lift": lift,
            "best_bl": best_bl_name,
        })

    # Sort
    if args.sort == "name":
        rows.sort(key=lambda r: r["name"])
    else:
        rows.sort(key=lambda r: r.get(args.sort) if r.get(args.sort) is not None else float("inf"),
                  reverse=(args.sort == "accuracy" or args.sort == "auc"))

    # Print
    max_name = max(len(r["name"]) for r in rows)
    max_src = min(40, max(len(r["sources"]) for r in rows))
    hdr = f"  {'Experiment':<{max_name}}  {'Val Sources':<{max_src}}  {'N':>7}  {'BCE':>8}  {'Acc':>8}  {'MAE':>8}  {'AUC':>8}  {'Lift':>8}"
    print(hdr)
    print(f"  {'-' * max_name}  {'-' * max_src}  {'-' * 7}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for r in rows:
        src_display = r["sources"][:max_src]
        print(f"  {r['name']:<{max_name}}  {src_display:<{max_src}}  {r['n']:>7}  {fmt(r['bce']):>8}  {fmt(r['accuracy']):>8}  {fmt(r['mae']):>8}  {fmt(r['auc']):>8}  {fmt_delta(r['lift']):>8}")

    # ===================================================================
    # 2. Baseline comparison table
    # ===================================================================
    print(f"\n{'=' * 90}")
    print("BASELINE COMPARISON (model accuracy vs running_avg accuracy)")
    print("=" * 90)

    bl_rows = []
    for exp in experiments:
        pred = exp["data"].get("prediction", {}).get("overall", {})
        baselines = exp["data"].get("baselines", {})
        row = {"name": exp["name"], "model": pred.get("accuracy")}
        for bl_name in ["global_mean", "running_avg", "last_outcome"]:
            bl_data = baselines.get(bl_name, {})
            row[bl_name] = bl_data.get("overall", {}).get("accuracy")
        bl_rows.append(row)

    if args.sort == "name":
        bl_rows.sort(key=lambda r: r["name"])
    else:
        bl_rows.sort(key=lambda r: r.get("model") if r.get("model") is not None else float("inf"),
                     reverse=True)

    print(f"  {'Experiment':<{max_name}}  {'Model':>8}  {'GlobalMean':>10}  {'RunAvg':>8}  {'LastOut':>8}")
    print(f"  {'-' * max_name}  {'-' * 8}  {'-' * 10}  {'-' * 8}  {'-' * 8}")
    for r in bl_rows:
        print(f"  {r['name']:<{max_name}}  {fmt(r['model']):>8}  {fmt(r.get('global_mean')):>10}  {fmt(r.get('running_avg')):>8}  {fmt(r.get('last_outcome')):>8}")

    # ===================================================================
    # 3. Per-source detail (optional)
    # ===================================================================
    if args.metric:
        metric = args.metric
        print(f"\n{'=' * 90}")
        print(f"PER-SOURCE DETAIL: {metric.upper()}")
        print("=" * 90)

        # Collect all sources across experiments
        all_sources = set()
        for exp in experiments:
            per_source = exp["data"].get("prediction", {}).get("per_source", {})
            all_sources.update(per_source.keys())
        all_sources = sorted(all_sources)

        if all_sources:
            max_src_col = max(len(s) for s in all_sources)
            header = f"  {'Source':<{max_src_col}}"
            for exp in experiments:
                header += f"  {exp['name'][:12]:>12}"
            print(header)
            print(f"  {'-' * max_src_col}" + "".join(f"  {'-' * 12}" for _ in experiments))

            for source in all_sources:
                line = f"  {source:<{max_src_col}}"
                for exp in experiments:
                    val = exp["data"].get("prediction", {}).get("per_source", {}).get(source, {}).get(metric)
                    line += f"  {fmt(val):>12}"
                print(line)

    # ===================================================================
    # 4. Forecast summary
    # ===================================================================
    print(f"\n{'=' * 90}")
    print("FORECAST ACCURACY (50% prefix)")
    print("=" * 90)

    fc_rows = []
    for exp in experiments:
        fc_data = exp["data"].get("forecast", {})
        fc_50 = fc_data.get("0.5", {})
        fc_rows.append({"name": exp["name"], "accuracy": fc_50.get("accuracy"), "bce": fc_50.get("bce")})

    if args.sort == "name":
        fc_rows.sort(key=lambda r: r["name"])
    else:
        fc_rows.sort(key=lambda r: r.get("accuracy") if r.get("accuracy") is not None else float("inf"),
                     reverse=True)

    print(f"  {'Experiment':<{max_name}}  {'Acc':>8}  {'BCE':>8}")
    print(f"  {'-' * max_name}  {'-' * 8}  {'-' * 8}")
    for r in fc_rows:
        print(f"  {r['name']:<{max_name}}  {fmt(r['accuracy']):>8}  {fmt(r['bce']):>8}")

    # ===================================================================
    # 5. CSV output (optional)
    # ===================================================================
    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment", "val_sources", "n", "bce", "accuracy", "mae", "auc", "lift_vs_best_baseline"])
            for r in rows:
                writer.writerow([r["name"], r["sources"], r["n"],
                                 r["bce"], r["accuracy"], r["mae"], r["auc"], r["lift"]])
        print(f"\nCSV saved to {args.output}")


if __name__ == "__main__":
    main()
