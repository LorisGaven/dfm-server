"""
Demo: test the /search endpoint on validation data.

Registers a learner with partial history, runs the GA search with increasing
generation counts to visualize convergence, and compares the optimized
curriculum against random baselines.

Usage (requires a running server):
    python search_demo.py --data checkpoint/val.jsonl --server-url http://localhost:8000
"""

import argparse
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dfm_server.client import DFMClient


def evaluate_curriculum(client, history_tasks, history_outcomes, curriculum, targets):
    """Apply a curriculum (autoregressive) and measure target competence.

    Returns (per_target_preds, mean_competence).
    """
    lid = f"_eval_{time.monotonic_ns()}"
    client.register(lid, tasks=history_tasks, outcomes=history_outcomes)
    forecast = client.forecast(lid, [curriculum])[0]
    for task, outcome in zip(curriculum, forecast):
        client.update(lid, task, outcome)
    preds = client.predict(lid, targets)
    client.delete(lid)
    return preds, sum(preds) / len(preds), forecast


def main():
    parser = argparse.ArgumentParser(description="Search endpoint demo")
    parser.add_argument("--data", required=True, help="Path to val.jsonl")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--learner-index", type=int, default=0)
    parser.add_argument("--split", type=float, default=0.5, help="History prefix fraction")
    parser.add_argument("--depth", type=int, default=64, help="Search sequence length")
    parser.add_argument("--n-targets", type=int, default=32)
    parser.add_argument("--n-candidates", type=int, default=0, help="0 = all tasks")
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--max-generations", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=None, help="Evaluate targets every N steps (default: end only)")
    parser.add_argument("--n-random-baselines", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="search_demo", help="Output prefix for plots")
    args = parser.parse_args()

    client = DFMClient(args.server_url)
    print(f"Server: {client.health()}")

    # ----- Load data -----
    with open(args.data) as f:
        all_learners = [json.loads(l) for l in f]

    server_tasks = set(client.tasks())
    all_tasks = set()
    for l in all_learners:
        all_tasks.update(l["tasks"])
    all_tasks = sorted(all_tasks & server_tasks)
    print(f"Learners: {len(all_learners)}, Unique tasks (in data & server): {len(all_tasks)}")

    # ----- Select learner, split history -----
    learner = all_learners[args.learner_index]
    T = len(learner["tasks"])
    split_idx = max(1, int(T * args.split))
    history_tasks = learner["tasks"][:split_idx]
    history_outcomes = learner["outcomes"][:split_idx]
    print(f"\nLearner: {learner['source']} ({T} steps, using first {split_idx} as history)")

    # ----- Select targets and candidates -----
    rng = random.Random(args.seed)
    targets = rng.sample(all_tasks, min(args.n_targets, len(all_tasks)))
    if args.n_candidates > 0 and args.n_candidates < len(all_tasks):
        candidates = rng.sample(all_tasks, args.n_candidates)
    else:
        candidates = all_tasks
    print(f"Targets: {len(targets)}, Candidates: {len(candidates)}, Depth: {args.depth}")

    # ----- Register learner -----
    client.register("search_demo", tasks=history_tasks, outcomes=history_outcomes)

    # ----- Baseline: target competence with no curriculum -----
    baseline_preds = client.predict("search_demo", targets)
    baseline_comp = sum(baseline_preds) / len(baseline_preds)
    print(f"\nBaseline target competence (no curriculum): {baseline_comp:.4f}")

    # ----- Fitness over generations -----
    milestones = sorted(
        {g for g in [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]
         if g <= args.max_generations} | {args.max_generations}
    )
    print(f"\nGeneration sweep: {milestones}")

    gen_fitness = []
    gen_times = []
    gen_sequences = []
    for g in milestones:
        t0 = time.time()
        result = client.search(
            "search_demo",
            target_tasks=targets,
            candidate_tasks=candidates,
            depth=args.depth,
            generations=g,
            population_size=args.population_size,
            eval_every=args.eval_every,
            seed=args.seed,
        )
        dt = time.time() - t0
        gen_fitness.append(result["best_fitness"])
        gen_times.append(dt)
        gen_sequences.append(result["best_sequence"])
        print(f"  gen={g:>3d}: fitness={result['best_fitness']:.4f}  ({dt:.2f}s)")

    # ----- Random baselines (gen=1, different seeds) -----
    print(f"\nRandom baselines ({args.n_random_baselines} seeds, gen=1):")
    random_fitnesses = []
    for i in range(args.n_random_baselines):
        r = client.search(
            "search_demo",
            target_tasks=targets,
            candidate_tasks=candidates,
            depth=args.depth,
            generations=1,
            population_size=args.population_size,
            eval_every=args.eval_every,
            seed=args.seed + 1000 + i,
        )
        random_fitnesses.append(r["best_fitness"])
    rnd_mean = sum(random_fitnesses) / len(random_fitnesses)
    rnd_min, rnd_max = min(random_fitnesses), max(random_fitnesses)
    print(f"  mean={rnd_mean:.4f}, min={rnd_min:.4f}, max={rnd_max:.4f}")

    # ----- Evaluate best vs random curriculum effect on targets -----
    best_seq = gen_sequences[-1]
    rnd_seq = gen_sequences[0]  # gen=1 = best random individual

    opt_preds, opt_comp, opt_forecast = evaluate_curriculum(
        client, history_tasks, history_outcomes, best_seq, targets
    )
    rnd_preds, rnd_comp, rnd_forecast = evaluate_curriculum(
        client, history_tasks, history_outcomes, rnd_seq, targets
    )

    print(f"\nTarget competence after curriculum:")
    print(f"  No curriculum:        {baseline_comp:.4f}")
    print(f"  Random (gen=1):       {rnd_comp:.4f}  (delta: {rnd_comp - baseline_comp:+.4f})")
    print(f"  Optimized (gen={args.max_generations:>2d}):   {opt_comp:.4f}  (delta: {opt_comp - baseline_comp:+.4f})")
    print(f"  Search fitness:       {gen_fitness[-1]:.4f}")

    # Cleanup
    client.delete("search_demo")

    # ----- Print best sequence -----
    print(f"\nBest sequence (gen={args.max_generations}):")
    for i, (task, p) in enumerate(zip(best_seq, opt_forecast)):
        short = task[:60] + "..." if len(task) > 60 else task
        print(f"  {i + 1:>2d}. [{p:.2f}] {short}")

    # ===================================================================
    # Plots
    # ===================================================================

    # --- Fig 1: Convergence + timing ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(milestones, gen_fitness, "o-", color="tab:blue", linewidth=2,
             label="GA best fitness", zorder=3)
    ax1.axhline(rnd_mean, color="tab:orange", linestyle="--", alpha=0.8,
                label=f"Random mean ({rnd_mean:.4f})")
    ax1.fill_between([milestones[0], milestones[-1]], rnd_min, rnd_max,
                     color="tab:orange", alpha=0.12, label="Random range")
    ax1.axhline(baseline_comp, color="tab:red", linestyle=":",
                label=f"No curriculum ({baseline_comp:.4f})")
    ax1.set_xlabel("Generations")
    ax1.set_ylabel("Fitness (mean target competence)")
    ax1.set_title("GA Convergence")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(milestones, gen_times, "s-", color="tab:green", linewidth=2)
    ax2.set_xlabel("Generations")
    ax2.set_ylabel("Wall time (s)")
    ax2.set_title("Search Latency")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"{learner['source']}  |  depth={args.depth}  pop={args.population_size}  "
        f"cand={len(candidates)}  tgt={len(targets)}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(f"{args.output}_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.output}_convergence.png")

    # --- Fig 2: Forecasted trajectory ---
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = list(range(1, args.depth + 1))
    ax.plot(steps, opt_forecast, "o-", color="tab:blue", linewidth=2,
            label=f"Optimized (gen={args.max_generations})")
    ax.plot(steps, rnd_forecast, "s--", color="tab:orange", linewidth=1.5,
            label="Random (gen=1)")
    ax.set_xlabel("Curriculum step")
    ax.set_ylabel("Predicted outcome probability")
    ax.set_title("Forecasted Outcomes Along Curriculum")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(steps)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.output}_trajectory.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {args.output}_trajectory.png")

    # --- Fig 3: Per-target competence comparison ---
    fig, ax = plt.subplots(figsize=(max(10, len(targets) * 0.5), 6))
    x = range(len(targets))
    w = 0.27
    ax.bar([i - w for i in x], baseline_preds, w, label="No curriculum",
           color="tab:red", alpha=0.7)
    ax.bar(list(x), rnd_preds, w, label="Random curriculum",
           color="tab:orange", alpha=0.7)
    ax.bar([i + w for i in x], opt_preds, w, label="Optimized curriculum",
           color="tab:blue", alpha=0.7)
    ax.set_xticks(list(x))
    labels = [(t[:25] + "..." if len(t) > 25 else t).replace("$", r"\$") for t in targets]
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Predicted competence")
    ax.set_title("Per-Target Competence: No Curriculum vs Random vs Optimized")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{args.output}_targets.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {args.output}_targets.png")


if __name__ == "__main__":
    main()
