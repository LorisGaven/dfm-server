"""Genetic algorithm search for optimal task sequences."""

import random

import torch

from .model import (
    BatchedKVCache,
    DFM,
    KVCache,
    embed_outcome_tokens,
    embed_task_token,
    predict_from_hiddens,
    transformer_forward,
)


@torch.no_grad()
def evaluate_population(
    model: DFM,
    kv_cache: KVCache,
    population_embs: torch.Tensor,
    target_embs: torch.Tensor,
    eval_steps: set[int],
) -> torch.Tensor:
    """Evaluate fitness of an entire population in a batched forward pass.

    Args:
        model: the DFM model
        kv_cache: learner's KV cache (not mutated)
        population_embs: (P, D, n_input) — pre-looked-up task embeddings for each individual
        target_embs: (T, n_input) — target task embeddings
        eval_steps: set of time steps (0-indexed) at which to evaluate targets

    Returns:
        (P,) fitness tensor — mean predicted competence on target tasks
    """
    P, D, _ = population_embs.shape
    T = target_embs.shape[0]

    # Fork learner cache for the population
    forecast_cache = BatchedKVCache(kv_cache, batch_size=P, extra_len=2 * D)

    competences = []

    for t in range(D):
        # Task token for step t
        step_embs = population_embs[:, t : t + 1, :]  # (P, 1, n_input)
        x = embed_task_token(model, step_embs)  # (P, 1, D_embd)
        hidden = transformer_forward(model, x, forecast_cache)  # (P, 1, D_embd)
        probs = predict_from_hiddens(model, hidden)  # (P,)

        # Outcome token (advance cache)
        x = embed_outcome_tokens(model, probs)  # (P, 1, D_embd)
        transformer_forward(model, x, forecast_cache)

        if t in eval_steps:
            # Fork for target evaluation: P sequences -> P*T sequences
            eval_cache = BatchedKVCache.fork(forecast_cache, fan_out=T, extra_len=1)

            # Prepare target embeddings: (P*T, 1, n_input)
            # target_embs is (T, n_input), expand to (P, T, n_input), reshape to (P*T, 1, n_input)
            target_batch = target_embs.unsqueeze(0).expand(P, T, -1).reshape(P * T, 1, -1)

            x = embed_task_token(model, target_batch)  # (P*T, 1, D_embd)
            hidden = transformer_forward(model, x, eval_cache)  # (P*T, 1, D_embd)
            target_probs = predict_from_hiddens(model, hidden)  # (P*T,)

            # Reshape to (P, T), mean over targets -> (P,) competence
            competence = target_probs.reshape(P, T).mean(dim=1)
            competences.append(competence)
            del eval_cache

    # Mean competence across eval points -> (P,)
    return torch.stack(competences).mean(dim=0)


def run_search(
    model: DFM,
    kv_cache: KVCache,
    candidate_indices: list[int],
    target_indices: list[int],
    candidate_names: list[str],
    emb_tensor: torch.Tensor,
    depth: int,
    population_size: int,
    generations: int,
    elite_count: int,
    tournament_size: int,
    crossover_rate: float,
    mutation_rate: float,
    eval_every: int | None,
    seed: int | None,
) -> tuple[list[str], float]:
    """Run genetic algorithm search for the optimal task sequence.

    Returns:
        (best_sequence, best_fitness) — task names and fitness score
    """
    rng = random.Random(seed)
    device = model.get_device()
    C = len(candidate_indices)
    P = population_size

    # Pre-compute embeddings
    candidate_idx_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=device)
    target_idx_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)
    candidate_embs = emb_tensor[candidate_idx_tensor]  # (C, n_input)
    target_embs = emb_tensor[target_idx_tensor]  # (T, n_input)

    # Determine eval steps
    if eval_every is None:
        eval_steps = {depth - 1}
    else:
        eval_steps = set(range(eval_every - 1, depth, eval_every))
        eval_steps.add(depth - 1)

    # Initialize random population: (P, D) indices into [0, C)
    population = torch.tensor(
        [[rng.randrange(C) for _ in range(depth)] for _ in range(P)],
        dtype=torch.long,
        device=device,
    )

    best_ever_seq = None
    best_ever_fitness = -1.0

    for gen in range(generations):
        # Look up embeddings: (P, D, n_input)
        pop_embs = candidate_embs[population]

        # Evaluate fitness
        fitness = evaluate_population(model, kv_cache, pop_embs, target_embs, eval_steps)

        # Track best-ever
        gen_best_idx = fitness.argmax().item()
        gen_best_fitness = fitness[gen_best_idx].item()
        if gen_best_fitness > best_ever_fitness:
            best_ever_fitness = gen_best_fitness
            best_ever_seq = population[gen_best_idx].clone()

        # Skip selection/crossover/mutation on the last generation
        if gen == generations - 1:
            break

        # Selection + crossover + mutation (CPU)
        fitness_list = fitness.tolist()
        pop_list = population.tolist()

        new_pop = []

        # Elitism: keep top individuals
        elite_indices = sorted(range(P), key=lambda i: fitness_list[i], reverse=True)[:elite_count]
        for idx in elite_indices:
            new_pop.append(pop_list[idx][:])

        # Fill the rest
        while len(new_pop) < P:
            # Tournament selection for two parents
            parent_a = _tournament_select(pop_list, fitness_list, tournament_size, rng)
            parent_b = _tournament_select(pop_list, fitness_list, tournament_size, rng)

            # Crossover
            if rng.random() < crossover_rate:
                child_a, child_b = _crossover(parent_a, parent_b, rng)
            else:
                child_a, child_b = parent_a[:], parent_b[:]

            # Mutation
            _mutate(child_a, C, mutation_rate, rng)
            _mutate(child_b, C, mutation_rate, rng)

            new_pop.append(child_a)
            if len(new_pop) < P:
                new_pop.append(child_b)

        population = torch.tensor(new_pop, dtype=torch.long, device=device)

    # Map best sequence to task names
    best_names = [candidate_names[i] for i in best_ever_seq.tolist()]
    return best_names, best_ever_fitness


def _tournament_select(
    pop: list[list[int]], fitness: list[float], k: int, rng: random.Random
) -> list[int]:
    contestants = rng.sample(range(len(pop)), min(k, len(pop)))
    best = max(contestants, key=lambda i: fitness[i])
    return pop[best]


def _crossover(a: list[int], b: list[int], rng: random.Random) -> tuple[list[int], list[int]]:
    if len(a) <= 1:
        return a[:], b[:]
    point = rng.randint(1, len(a) - 1)
    return a[:point] + b[point:], b[:point] + a[point:]


def _mutate(individual: list[int], n_candidates: int, rate: float, rng: random.Random):
    for i in range(len(individual)):
        if rng.random() < rate:
            individual[i] = rng.randrange(n_candidates)
