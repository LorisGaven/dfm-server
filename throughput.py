"""
Exhaustive throughput & memory limit benchmark for the DFM inference server.

Sweeps (prefix_length, horizon, batch_size) until OOM to find the maximum
number of parallel curriculum sequences your GPU can handle.

Usage:
    python throughput.py [--server-url http://localhost:8000]
"""

import argparse
import random
import time

import requests

from dfm_server.client import DFMClient

# ---------------------------------------------------------------------------
# Test grid
# ---------------------------------------------------------------------------

PREFIX_LENGTHS = [0, 50, 100, 200, 500]
HORIZON_LENGTHS = [10, 50, 100, 200]
# Batch sizes: start small, double until OOM
MAX_BATCH_POWER = 12  # up to 2^12 = 4096

WARMUP = 1
REPEATS = 3


def bench(fn, warmup=WARMUP, repeats=REPEATS):
    """Run fn, return median time in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def try_forecast(client, lid, prefix_tasks, prefix_outcomes, seqs):
    """Try a register+forecast+delete cycle. Returns (success, time_s) or (False, None) on OOM."""
    try:
        client.register(lid, tasks=prefix_tasks, outcomes=prefix_outcomes)
    except requests.HTTPError as e:
        # OOM during register (prefix prefill)
        try:
            client.delete(lid)
        except Exception:
            pass
        client.gc()
        return False, None

    try:
        t0 = time.perf_counter()
        client.forecast(lid, seqs)
        t = time.perf_counter() - t0
    except requests.HTTPError:
        # OOM during forecast
        try:
            client.delete(lid)
        except Exception:
            pass
        client.gc()
        return False, None

    client.delete(lid)
    return True, t


def estimate_kv_memory_mb(n_layer, n_kv_head, head_dim, prefix_tokens, horizon, batch_size):
    """Estimate BatchedKVCache memory in MB (bf16 = 2 bytes per element)."""
    total_tokens = prefix_tokens + 2 * horizon  # prefix interleaved + forecast interleaved
    # k + v, each: (n_layers, S, n_kv_head, total_tokens, head_dim)
    elements = 2 * n_layer * batch_size * n_kv_head * total_tokens * head_dim
    return elements * 2 / (1024 * 1024)  # bf16


def main():
    parser = argparse.ArgumentParser(description="DFM exhaustive throughput benchmark")
    parser.add_argument("--server-url", default="http://localhost:8000")
    args = parser.parse_args()

    client = DFMClient(args.server_url)
    health = client.health()
    print(f"Server OK: {health['learner_count']} learners")

    all_tasks = client.tasks()
    cfg = client.config()
    print(f"Tasks: {len(all_tasks)}  |  Model: {cfg['n_layer']}L, {cfg['n_embd']}D, {cfg['n_head']}Q/{cfg['n_kv_head']}KV")

    head_dim = cfg["n_embd"] // cfg["n_head"]
    n_layer = cfg["n_layer"]
    n_kv_head = cfg["n_kv_head"]

    rng = random.Random(42)

    def rand_tasks(n):
        return rng.choices(all_tasks, k=n)

    def rand_outcomes(n):
        return [float(rng.random() > 0.5) for _ in range(n)]

    def make_sequences(S, L):
        return [rand_tasks(L) for _ in range(S)]

    # ===================================================================
    # Sweep: for each (prefix, horizon), find max batch size
    # ===================================================================
    print()
    print("=" * 80)
    print("EXHAUSTIVE SWEEP — finding max batch size before OOM")
    print("=" * 80)

    # results[prefix][horizon] = list of (S, time_s, steps_s) up to OOM
    results = {}
    max_batch_for = {}  # (prefix, horizon) -> max S

    for prefix_n in PREFIX_LENGTHS:
        prefix_tasks = rand_tasks(prefix_n) if prefix_n > 0 else None
        prefix_outcomes = rand_outcomes(prefix_n) if prefix_n > 0 else None
        prefix_tokens = 2 * prefix_n + 2 if prefix_n > 0 else 2  # interleaved + BOS pair

        for horizon in HORIZON_LENGTHS:
            key = (prefix_n, horizon)
            results[key] = []

            print(f"\n  prefix={prefix_n:>4}, L={horizon:>3}  ", end="", flush=True)

            for power in range(MAX_BATCH_POWER + 1):
                S = 2 ** power
                seqs = make_sequences(S, horizon)
                lid = f"tp_{prefix_n}_{horizon}_{S}"

                mem_mb = estimate_kv_memory_mb(n_layer, n_kv_head, head_dim, prefix_tokens, horizon, S)

                # Warm up
                ok, _ = try_forecast(client, lid, prefix_tasks, prefix_outcomes, seqs)
                if not ok:
                    print(f"OOM at S={S} (~{mem_mb:.0f}MB)", flush=True)
                    break

                # Benchmark
                times = []
                for _ in range(REPEATS):
                    ok, t = try_forecast(client, lid, prefix_tasks, prefix_outcomes, seqs)
                    if not ok:
                        break
                    times.append(t)

                if not times:
                    print(f"OOM at S={S} (~{mem_mb:.0f}MB)", flush=True)
                    break

                times.sort()
                t = times[len(times) // 2]
                total_steps = S * horizon
                steps_s = total_steps / t
                results[key].append((S, t, steps_s, mem_mb))
                max_batch_for[key] = S

                print(f"S={S}({steps_s:.0f}st/s) ", end="", flush=True)
            else:
                print(f"no OOM up to S={2**MAX_BATCH_POWER}", flush=True)

    # ===================================================================
    # Summary table: max batch size
    # ===================================================================
    print()
    print("=" * 80)
    print("MAX BATCH SIZE (sequences in parallel)")
    print("=" * 80)

    # Header
    header = f"  {'':>10}"
    for L in HORIZON_LENGTHS:
        header += f"  {'L='+str(L):>12}"
    print(header)
    print(f"  {'':>10}" + "  ".join(["-" * 12] * len(HORIZON_LENGTHS)))

    for prefix_n in PREFIX_LENGTHS:
        row = f"  {'ctx='+str(prefix_n):>10}"
        for L in HORIZON_LENGTHS:
            key = (prefix_n, L)
            if key in max_batch_for:
                S = max_batch_for[key]
                row += f"  {S:>12}"
            else:
                row += f"  {'OOM@1':>12}"
        print(row)

    # ===================================================================
    # Summary table: peak throughput (steps/s)
    # ===================================================================
    print()
    print("=" * 80)
    print("PEAK THROUGHPUT (steps/s at max viable batch size)")
    print("=" * 80)

    header = f"  {'':>10}"
    for L in HORIZON_LENGTHS:
        header += f"  {'L='+str(L):>12}"
    print(header)
    print(f"  {'':>10}" + "  ".join(["-" * 12] * len(HORIZON_LENGTHS)))

    for prefix_n in PREFIX_LENGTHS:
        row = f"  {'ctx='+str(prefix_n):>10}"
        for L in HORIZON_LENGTHS:
            key = (prefix_n, L)
            if key in results and results[key]:
                best = results[key][-1]  # last viable = highest batch
                row += f"  {best[2]:>12.0f}"
            else:
                row += f"  {'-':>12}"
        print(row)

    # ===================================================================
    # Summary table: estimated KV memory at max batch
    # ===================================================================
    print()
    print("=" * 80)
    print("KV CACHE MEMORY AT MAX BATCH (MB)")
    print("=" * 80)

    header = f"  {'':>10}"
    for L in HORIZON_LENGTHS:
        header += f"  {'L='+str(L):>12}"
    print(header)
    print(f"  {'':>10}" + "  ".join(["-" * 12] * len(HORIZON_LENGTHS)))

    for prefix_n in PREFIX_LENGTHS:
        row = f"  {'ctx='+str(prefix_n):>10}"
        for L in HORIZON_LENGTHS:
            key = (prefix_n, L)
            if key in results and results[key]:
                best = results[key][-1]
                row += f"  {best[3]:>10.0f}MB"
            else:
                row += f"  {'-':>12}"
        print(row)

    # ===================================================================
    # Sequential baseline
    # ===================================================================
    print()
    print("=" * 80)
    print("SEQUENTIAL BASELINE (predict+update)")
    print("=" * 80)

    n_steps = 50
    seq_tasks = rand_tasks(n_steps)
    seq_outcomes = rand_outcomes(n_steps)
    lid = "tp_seq"

    def run_sequential():
        client.register(lid)
        for task, outcome in zip(seq_tasks, seq_outcomes):
            client.predict(lid, [task])
            client.update(lid, task, outcome)
        client.delete(lid)

    t_seq = bench(run_sequential)
    seq_steps_s = n_steps / t_seq
    print(f"  {n_steps} predict+update cycles: {t_seq*1000:.1f}ms ({seq_steps_s:.0f} steps/s)")

    # ===================================================================
    # Plot
    # ===================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # --- Plot 1: Throughput vs batch size (one line per config) ---
        ax = axes[0]
        cmap = plt.cm.viridis
        configs = [(p, l) for p in PREFIX_LENGTHS for l in HORIZON_LENGTHS if results.get((p, l))]
        for i, (p, l) in enumerate(configs):
            data = results[(p, l)]
            ss = [d[0] for d in data]
            throughputs = [d[2] for d in data]
            color = cmap(i / max(1, len(configs) - 1))
            ax.plot(ss, throughputs, "o-", color=color, markersize=4, label=f"ctx={p},L={l}")
        ax.axhline(seq_steps_s, color="tab:red", linestyle="--", alpha=0.6, linewidth=2, label=f"Sequential: {seq_steps_s:.0f}")
        ax.set_xlabel("Batch size (S)")
        ax.set_ylabel("Steps/s")
        ax.set_title("Throughput vs Batch Size")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

        # --- Plot 2: Max batch size heatmap ---
        ax = axes[1]
        grid = np.zeros((len(PREFIX_LENGTHS), len(HORIZON_LENGTHS)))
        for i, p in enumerate(PREFIX_LENGTHS):
            for j, l in enumerate(HORIZON_LENGTHS):
                grid[i, j] = max_batch_for.get((p, l), 0)
        im = ax.imshow(grid, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(HORIZON_LENGTHS)))
        ax.set_xticklabels([str(l) for l in HORIZON_LENGTHS])
        ax.set_yticks(range(len(PREFIX_LENGTHS)))
        ax.set_yticklabels([str(p) for p in PREFIX_LENGTHS])
        ax.set_xlabel("Horizon (L)")
        ax.set_ylabel("Prefix length")
        ax.set_title("Max Batch Size Before OOM")
        for i in range(len(PREFIX_LENGTHS)):
            for j in range(len(HORIZON_LENGTHS)):
                v = int(grid[i, j])
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="white" if v > grid.max() * 0.6 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)

        # --- Plot 3: Peak throughput heatmap ---
        ax = axes[2]
        grid_tp = np.zeros((len(PREFIX_LENGTHS), len(HORIZON_LENGTHS)))
        for i, p in enumerate(PREFIX_LENGTHS):
            for j, l in enumerate(HORIZON_LENGTHS):
                key = (p, l)
                if key in results and results[key]:
                    grid_tp[i, j] = results[key][-1][2]
        im = ax.imshow(grid_tp, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(HORIZON_LENGTHS)))
        ax.set_xticklabels([str(l) for l in HORIZON_LENGTHS])
        ax.set_yticks(range(len(PREFIX_LENGTHS)))
        ax.set_yticklabels([str(p) for p in PREFIX_LENGTHS])
        ax.set_xlabel("Horizon (L)")
        ax.set_ylabel("Prefix length")
        ax.set_title("Peak Throughput (steps/s)")
        for i in range(len(PREFIX_LENGTHS)):
            for j in range(len(HORIZON_LENGTHS)):
                v = int(grid_tp[i, j])
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if v > grid_tp.max() * 0.6 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)

        fig.tight_layout()
        plot_path = "throughput.png"
        fig.savefig(plot_path, dpi=150)
        print(f"\n  Plot saved to {plot_path}")
        plt.close(fig)
    except ImportError:
        print("\n  (matplotlib not installed — skipping plot)")

    # ===================================================================
    # Final summary
    # ===================================================================
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if max_batch_for:
        best_key = max(max_batch_for, key=lambda k: results[k][-1][2] if results[k] else 0)
        best = results[best_key][-1]
        print(f"  Peak throughput:  {best[2]:.0f} steps/s (ctx={best_key[0]}, L={best_key[1]}, S={best[0]})")
        print(f"  Sequential:       {seq_steps_s:.0f} steps/s")
        print(f"  Max speedup:      {best[2]/seq_steps_s:.0f}x")

        # Largest batch across all configs
        biggest = max(max_batch_for.values())
        print(f"  Largest batch:    S={biggest}")

        # Most constrained config
        smallest_key = min(max_batch_for, key=max_batch_for.get)
        print(f"  Most constrained: ctx={smallest_key[0]}, L={smallest_key[1]} → max S={max_batch_for[smallest_key]}")


if __name__ == "__main__":
    main()
