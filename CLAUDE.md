# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

DFM-Server is a FastAPI inference server for the Developmental Foundation Model (DFM) — a universal outcome predictor. Given a learner's history of (task, outcome) pairs, it predicts the outcome on the next task using an HL-Gauss (Histogram Loss with Gaussian smoothing) prediction head. Supports learner registration, outcome prediction, autoregressive forecasting, and per-learner KV-cache management.

The model is trained in the sibling repository `../dfm-training/` (has its own `CLAUDE.md` with full details on the training pipeline, data collection, and model architecture).

## Common Commands

### Install
```bash
conda run -n dfm-server pip install -e .
```

### Run server
```bash
conda run -n dfm-server dfm-server --checkpoint checkpoint/ckpt.pt --embeddings embeddings.pt --device cuda
```
Or via environment variables:
```bash
conda run -n dfm-server env DFM_CHECKPOINT_PATH=checkpoint/ckpt.pt DFM_EMBEDDINGS_PATH=embeddings.pt python -m dfm_server.server
```

### Lint
```bash
ruff check .
ruff format .
```

### SLURM (Jean Zay cluster)
Benchmarking SLURM scripts in `slurm/`: `benchmark.slurm` runs the server + benchmark on a checkpoint directory. Generalization experiment benchmarks: `bench_baseline.slurm`, `bench_scale_*.slurm`, `bench_hold_*.slurm`, `bench_cross_*.slurm`.

## Architecture

### Module layout
- **`dfm_server/server.py`** — FastAPI app with lifespan model loading, all HTTP endpoints, live request logging via Rich (ANSI box redrawn per request)
- **`dfm_server/model.py`** — DFM transformer model (copied from dfm-training, inference-only) with HL-Gauss prediction head, KVCache (single-learner), BatchedKVCache (forked for forecast), and token-level inference helpers
- **`dfm_server/schemas.py`** — Pydantic request/response models
- **`dfm_server/client.py`** — Python HTTP client wrapper (`DFMClient`)

### Request flow
1. **Register** (`POST /learners`) — creates a KVCache, optionally prefills with history
2. **Predict** (`POST /predict`) — runs forward pass at task positions, saves/restores cache position (non-destructive)
3. **Update** (`POST /update`) — embeds task+outcome tokens and advances the cache (destructive)
4. **Forecast** (`POST /forecast`) — forks prefix into a BatchedKVCache, runs autoregressive prediction over multiple task sequences in parallel (non-destructive to the learner's primary cache). Uses Beta sampling for outcome feedback during autoregressive rollout.

### Additional endpoints
- `DELETE /learners/{id}` — delete a learner
- `GET /health` — status and learner count
- `GET /tasks` — list all known task strings
- `GET /config` — model configuration
- `POST /gc` — free cached CUDA memory

### Startup: loading pre-computed embeddings
At startup, the server loads an `embeddings.pt` file containing `{"tasks": [...], "embeddings": Tensor(N, D)}` (pre-computed by `dfm_training.embed_tasks`). It builds a `task_to_idx` mapping from the tasks list and constructs an embedding tensor with a BOS zero row prepended (row 0 = BOS, rows 1+ = pre-computed). No encoder is loaded — embeddings are fully pre-computed.

### Interleaved sequence format
Input is `[task₀, out₀, task₁, out₁, ...]` (2*T tokens for T task-outcome pairs). Predictions are extracted from hidden states at task positions. Type embeddings distinguish tasks (0) from outcomes (1).

### KV cache design
- `KVCache` — per-learner, shape `(n_layers, 1, n_kv_head, max_len, head_dim)`. Predict saves/restores position; Update advances it.
- `BatchedKVCache` — forked from a learner's prefix for batched forecast. Views into the source prefix (zero-copy), allocates only the per-batch suffix. Also supports `fork()` class method to re-fork a BatchedKVCache into a larger batch with additional slots.

### Prediction model
The prediction head outputs `n_bins` logits over histogram bins spanning [0, 1] (HL-Gauss):
- **Predict endpoint**: returns expected value = `sum(bin_centers * softmax(logits))`.
- **Forecast endpoint**: returns expected value as predictions, but feeds categorically-sampled bin centers as autoregressive feedback to capture uncertainty in multi-step rollouts.

### Key invariants
- **BOS convention**: task index 0 = zeros, outcome = -1.0. Learned BOS embeddings replace these in the embedding layer.
- **Prefill-vs-incremental consistency**: registering with history then predicting must match register-empty + sequential updates + predict (tolerance < 1e-3 due to bf16).
- **Non-destructiveness**: Predict and Forecast never mutate the learner's primary cache state.
- **Explicit bf16**: no autocast; tensors are explicitly cast to bfloat16.
- **MAX_SEQ_LEN = 32768** interleaved tokens per learner.

### Model architecture
Transformer with rotary embeddings, QK norm, ReLU² MLP, Multi-Query Attention (n_head Q, n_kv_head KV). Prediction head outputs `n_bins` logits over histogram bins (HL-Gauss). Expected value via weighted sum of bin centers; sampling via categorical distribution.
