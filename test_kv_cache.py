"""Test that KV cache implementations produce the same output as naive forward pass."""

import torch

from dfm_server.model import (
    BatchedKVCache,
    DFM,
    DFMConfig,
    KVCache,
    norm,
    transformer_forward,
)

# Small model config for fast testing
CONFIG = DFMConfig(
    block_size=256,
    n_layer=3,
    n_head=4,
    n_kv_head=2,
    n_embd=64,
    n_input=32,
)

DTYPE = torch.bfloat16
DEVICE = "cuda"

# Tolerance: on GPU, SDPA is numerically consistent across sequence lengths,
# so naive and cached outputs should match exactly (0.0 diff in bf16).
# We use a small epsilon to guard against any future regressions.
ATOL = 1e-3


def make_model():
    """Create a small DFM model for testing."""
    model = DFM(CONFIG)
    model.to(device=DEVICE, dtype=DTYPE)
    model.eval()
    return model


def naive_forward(model, x):
    """Full-sequence forward pass without KV cache. Returns hidden states (B, T, D)."""
    T = x.size(1)
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    h = norm(x)
    for block in model.transformer.h:
        h = block(h, cos_sin, kv_cache=None)
    h = norm(h)
    return h


def random_embeddings(B, T):
    """Generate random already-embedded tokens (B, T, n_embd)."""
    return torch.randn(B, T, CONFIG.n_embd, dtype=DTYPE, device=DEVICE)


def test_kv_cache_basic():
    """KVCache: prefill C tokens, then forward 1 token at a time.
    Compare hidden states at each position with naive full-sequence forward.
    """
    print("=" * 60)
    print("TEST: KVCache basic (prefill + incremental)")
    print("=" * 60)

    model = make_model()
    C = 8  # context length
    N = 4  # incremental tokens

    x_full = random_embeddings(1, C + N)

    # Naive: full forward
    with torch.no_grad():
        h_naive = naive_forward(model, x_full)  # (1, C+N, D)

    # Incremental: prefill C, then 1 token at a time
    kv_cache = KVCache(CONFIG, max_len=256, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        h_prefix = transformer_forward(model, x_full[:, :C, :], kv_cache)
        incremental_hiddens = [h_prefix]
        for i in range(N):
            h_i = transformer_forward(model, x_full[:, C + i : C + i + 1, :], kv_cache)
            incremental_hiddens.append(h_i)
        h_cache = torch.cat(incremental_hiddens, dim=1)  # (1, C+N, D)

    diff = (h_naive - h_cache).abs().max().item()
    print(f"  Max abs diff: {diff:.6f} (atol={ATOL})")
    assert diff < ATOL, f"KVCache output mismatch: {diff} >= {ATOL}"
    print("  PASSED\n")


def test_kv_cache_chunk_prefill():
    """KVCache: prefill with a chunk > 1 token, then add more chunks.
    Ensures the chunk attention mask (prefix + causal within chunk) is correct.
    """
    print("=" * 60)
    print("TEST: KVCache chunk prefill")
    print("=" * 60)

    model = make_model()
    x_full = random_embeddings(1, 12)

    with torch.no_grad():
        h_naive = naive_forward(model, x_full)

    # Chunk: 5 + 4 + 3
    kv_cache = KVCache(CONFIG, max_len=256, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        h1 = transformer_forward(model, x_full[:, :5, :], kv_cache)
        h2 = transformer_forward(model, x_full[:, 5:9, :], kv_cache)
        h3 = transformer_forward(model, x_full[:, 9:12, :], kv_cache)
        h_cache = torch.cat([h1, h2, h3], dim=1)

    diff = (h_naive - h_cache).abs().max().item()
    print(f"  Max abs diff: {diff:.6f} (atol={ATOL})")
    assert diff < ATOL, f"Chunk prefill mismatch: {diff} >= {ATOL}"
    print("  PASSED\n")


def test_batched_kv_cache():
    """BatchedKVCache: fork from KVCache, forward S different suffix sequences.
    Compare each with a naive forward of prefix+suffix[s].
    """
    print("=" * 60)
    print("TEST: BatchedKVCache (fork from KVCache)")
    print("=" * 60)

    model = make_model()
    C = 6  # shared context
    L = 4  # per-sequence suffix
    S = 3  # batch size

    x_prefix = random_embeddings(1, C)
    x_suffixes = random_embeddings(S, L)  # different suffix per sequence

    # Naive: for each sequence, full forward of prefix + suffix
    naive_hiddens = []
    with torch.no_grad():
        for s in range(S):
            x_full = torch.cat([x_prefix, x_suffixes[s : s + 1]], dim=1)  # (1, C+L, D)
            h = naive_forward(model, x_full)
            naive_hiddens.append(h[:, C:, :])  # only suffix positions
    h_naive = torch.cat(naive_hiddens, dim=0)  # (S, L, D)

    # Batched: prefill prefix into KVCache, fork to BatchedKVCache, forward suffixes
    kv_cache = KVCache(CONFIG, max_len=256, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        transformer_forward(model, x_prefix, kv_cache)
        batched_cache = BatchedKVCache(kv_cache, batch_size=S, extra_len=L)
        h_batched = transformer_forward(model, x_suffixes, batched_cache)  # (S, L, D)

    diff = (h_naive - h_batched).abs().max().item()
    print(f"  Max abs diff: {diff:.6f} (atol={ATOL})")
    assert diff < ATOL, f"BatchedKVCache mismatch: {diff} >= {ATOL}"
    print("  PASSED\n")


def test_batched_fork():
    """BatchedKVCache.fork: fork S -> S*T, forward 1 target token each.
    Compare each (s, t) with naive forward of prefix + curriculum[s] + target[t].
    """
    print("=" * 60)
    print("TEST: BatchedKVCache.fork (S -> S*T)")
    print("=" * 60)

    model = make_model()
    C = 5  # shared context
    L = 3  # curriculum length
    S = 2  # number of curricula
    T = 4  # number of targets

    x_prefix = random_embeddings(1, C)
    x_curricula = random_embeddings(S, L)  # different curriculum per sequence
    x_targets = random_embeddings(1, T)  # shared targets

    # Naive: for each (s, t), full forward of prefix + curriculum[s] + target[t]
    naive_preds = torch.zeros(S, T, device=DEVICE)
    with torch.no_grad():
        for s in range(S):
            for t in range(T):
                x_full = torch.cat(
                    [x_prefix, x_curricula[s : s + 1], x_targets[:, t : t + 1]],
                    dim=1,
                )  # (1, C+L+1, D)
                h = naive_forward(model, x_full)
                # Prediction at last position
                pred = model.prediction_head(h[:, -1:, :]).squeeze(-1).float()
                naive_preds[s, t] = torch.sigmoid(pred).item()

    # Cached: prefill prefix, fork to S for curriculum, fork to S*T for targets
    kv_cache = KVCache(CONFIG, max_len=256, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        transformer_forward(model, x_prefix, kv_cache)

        # Fork to S, forward curriculum
        batched_cache = BatchedKVCache(kv_cache, batch_size=S, extra_len=L)
        transformer_forward(model, x_curricula, batched_cache)

        # Fork S -> S*T for targets (this is the three-level cache)
        target_cache = BatchedKVCache.fork(batched_cache, fan_out=T, extra_len=1)

        # Build target tokens: (S*T, 1, D)
        target_x = x_targets.expand(S, -1, -1).reshape(S * T, 1, -1)
        hidden = transformer_forward(model, target_x, target_cache)  # (S*T, 1, D)
        cached_preds = torch.sigmoid(
            model.prediction_head(hidden).squeeze(-1).float()
        ).reshape(S, T)

    diff = (naive_preds - cached_preds).abs().max().item()
    print(f"  Max abs diff: {diff:.6f} (atol={ATOL})")
    assert diff < ATOL, f"BatchedKVCache.fork mismatch: {diff} >= {ATOL}"

    # Also verify hidden states directly (not just predictions)
    # Redo to check hidden state at target positions
    naive_target_hiddens = torch.zeros(S, T, CONFIG.n_embd, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        for s in range(S):
            for t in range(T):
                x_full = torch.cat(
                    [x_prefix, x_curricula[s : s + 1], x_targets[:, t : t + 1]],
                    dim=1,
                )
                h = naive_forward(model, x_full)
                naive_target_hiddens[s, t] = h[0, -1, :]

    cached_target_hiddens = hidden.reshape(S, T, CONFIG.n_embd)
    h_diff = (naive_target_hiddens - cached_target_hiddens).abs().max().item()
    print(f"  Max abs hidden diff: {h_diff:.6f} (atol={ATOL})")
    assert h_diff < ATOL, f"Hidden state mismatch: {h_diff} >= {ATOL}"
    print("  PASSED\n")


def test_fork_no_curriculum():
    """Fork directly from KVCache to S*T (no curriculum, L=0).
    This tests the path where BatchedKVCache has no mid level.
    """
    print("=" * 60)
    print("TEST: Direct fork (no curriculum, targets only)")
    print("=" * 60)

    model = make_model()
    C = 6
    T = 5

    x_prefix = random_embeddings(1, C)
    x_targets = random_embeddings(1, T)

    # Naive: for each target, full forward of prefix + target[t]
    naive_preds = torch.zeros(T, device=DEVICE)
    with torch.no_grad():
        for t in range(T):
            x_full = torch.cat([x_prefix, x_targets[:, t : t + 1]], dim=1)
            h = naive_forward(model, x_full)
            pred = model.prediction_head(h[:, -1:, :]).squeeze(-1).float()
            naive_preds[t] = torch.sigmoid(pred).item()

    # Cached: prefill prefix, fork to T, forward each target independently
    kv_cache = KVCache(CONFIG, max_len=256, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        transformer_forward(model, x_prefix, kv_cache)
        batched_cache = BatchedKVCache(kv_cache, batch_size=T, extra_len=1)
        target_x = x_targets.squeeze(0).unsqueeze(1)  # (T, 1, D)
        hidden = transformer_forward(model, target_x, batched_cache)  # (T, 1, D)
        cached_preds = torch.sigmoid(model.prediction_head(hidden).float()).view(
            T
        )  # (T,)

    diff = (naive_preds - cached_preds).abs().max().item()
    print(f"  Max abs diff: {diff:.6f} (atol={ATOL})")
    assert diff < ATOL, f"Direct fork mismatch: {diff} >= {ATOL}"
    print("  PASSED\n")


def test_target_independence():
    """Verify that target predictions are independent of each other.
    Predicting target A alone should give the same result as predicting
    targets A, B, C together (each in their own forked cache slot).
    """
    print("=" * 60)
    print("TEST: Target independence (order doesn't matter)")
    print("=" * 60)

    model = make_model()
    C = 6
    L = 3
    T = 4

    x_prefix = random_embeddings(1, C)
    x_curriculum = random_embeddings(1, L)
    x_targets = random_embeddings(1, T)

    with torch.no_grad():
        # Run all T targets together
        kv_cache = KVCache(CONFIG, max_len=256, device=DEVICE, dtype=DTYPE)
        transformer_forward(model, x_prefix, kv_cache)
        batched_cache = BatchedKVCache(kv_cache, batch_size=1, extra_len=L)
        transformer_forward(model, x_curriculum, batched_cache)
        target_cache = BatchedKVCache.fork(batched_cache, fan_out=T, extra_len=1)
        target_x = x_targets.reshape(T, 1, -1)
        hidden_all = transformer_forward(model, target_x, target_cache)  # (T, 1, D)
        preds_all = torch.sigmoid(model.prediction_head(hidden_all).float()).view(
            T
        )  # (T,)

        # Run each target individually
        preds_individual = torch.zeros(T, device=DEVICE)
        for t in range(T):
            kv_cache_t = KVCache(CONFIG, max_len=256, device=DEVICE, dtype=DTYPE)
            transformer_forward(model, x_prefix, kv_cache_t)
            batched_t = BatchedKVCache(kv_cache_t, batch_size=1, extra_len=L)
            transformer_forward(model, x_curriculum, batched_t)
            target_cache_t = BatchedKVCache.fork(batched_t, fan_out=1, extra_len=1)
            hidden_t = transformer_forward(
                model, x_targets[:, t : t + 1, :], target_cache_t
            )
            preds_individual[t] = torch.sigmoid(
                model.prediction_head(hidden_t).squeeze(-1).float()
            ).item()

    diff = (preds_all - preds_individual).abs().max().item()
    print(f"  Max abs diff: {diff:.6f} (atol={ATOL})")
    assert diff < ATOL, f"Target independence violated: {diff} >= {ATOL}"
    print("  PASSED\n")


if __name__ == "__main__":
    test_kv_cache_basic()
    test_kv_cache_chunk_prefill()
    test_batched_kv_cache()
    test_batched_fork()
    test_fork_no_curriculum()
    test_target_independence()
    print("All KV cache tests passed!")
