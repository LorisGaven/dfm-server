# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

DFM-Server is a FastAPI inference server for the Developmental Foundation Model (DFM) — a universal outcome predictor. Given a learner's history of (task, outcome) pairs, it predicts the outcome on the next task. Supports learner registration, outcome prediction, autoregressive forecasting, and per-learner KV-cache management.

The model is trained in the sibling repository `../dfm-training/` (has its own `CLAUDE.md` with full details on the training pipeline, data collection, and model architecture).

## Common Commands

### Install
```bash
conda run -n dfm-server pip install -e .
```

### Run server
```bash
conda run -n dfm-server dfm-server --checkpoint checkpoint/ckpt.pt --embeddings checkpoint/embeddings.pt --device cuda
```
Or via environment variables:
```bash
conda run -n dfm-server env DFM_CHECKPOINT_PATH=checkpoint/ckpt.pt DFM_EMBEDDINGS_PATH=checkpoint/embeddings.pt python -m dfm_server.server
```

### Run tests
```bash
conda run -n dfm-server python tests/test_server.py
```
No pytest — uses a custom runner with plain assertions. Tests use a synthetic tiny model and FastAPI's TestClient (no running server needed).

### Lint
```bash
ruff check .
ruff format .
```

### Benchmark (requires running server)
```bash
python benchmark.py --data checkpoint/val.jsonl --server-url http://localhost:8000
python throughput.py --server-url http://localhost:8000
```

## Architecture

### Module layout
- **`dfm_server/server.py`** — FastAPI app with lifespan model loading, all HTTP endpoints, live request logging via Rich
- **`dfm_server/model.py`** — DFM transformer model, KVCache (single-learner), BatchedKVCache (forked for forecast), and token-level inference helpers
- **`dfm_server/schemas.py`** — Pydantic request/response models
- **`dfm_server/client.py`** — Python HTTP client wrapper

### Request flow
1. **Register** (`POST /learners`) — creates a KVCache, optionally prefills with history
2. **Predict** (`POST /predict`) — runs forward pass at task positions, saves/restores cache position (non-destructive)
3. **Update** (`POST /update`) — embeds task+outcome tokens and advances the cache (destructive)
4. **Forecast** (`POST /forecast`) — forks prefix into a BatchedKVCache, runs autoregressive prediction over multiple task sequences in parallel (non-destructive to the learner's primary cache)

### Interleaved sequence format
Input is `[task₀, out₀, task₁, out₁, ...]` (2*T tokens for T task-outcome pairs). Predictions are extracted from hidden states at task positions. Type embeddings distinguish tasks (0) from outcomes (1).

### KV cache design
- `KVCache` — per-learner, shape `(n_layers, 1, n_kv_head, max_len, head_dim)`. Predict saves/restores position; Update advances it.
- `BatchedKVCache` — forked from a learner's prefix for batched forecast. Broadcasts prefix to batch dimension, allocates extra slots for the forecast horizon.

### Key invariants
- **BOS convention**: task index 0 = zeros, outcome = -1.0. Learned BOS embeddings replace these in the embedding layer.
- **Prefill-vs-incremental consistency**: registering with history then predicting must match register-empty + sequential updates + predict (tolerance < 1e-3 due to bf16).
- **Non-destructiveness**: Predict and Forecast never mutate the learner's primary cache state.
- **Explicit bf16**: no autocast; tensors are explicitly cast to bfloat16.
- **MAX_SEQ_LEN = 4096** interleaved tokens per learner.

### Model architecture
Transformer with rotary embeddings, QK norm, ReLU² MLP, Multi-Query Attention (n_head Q, n_kv_head KV). Output head applies softcap (`15.0 * tanh(logit / 15.0)`) then sigmoid for probability in [0, 1].
