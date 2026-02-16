# dfm-server

Inference server for models trained with [dfm-training](https://github.com/…/dfm-training). The server exposes a set of HTTP endpoints that support a range of downstream applications — benchmark performance prediction, curriculum design, hyperparameter optimization, and more.

## Core capabilities

The API supports operations such as:

- **Registering learners** — either with a prefilled history of tasks/outcomes or from scratch (BOS-only).
- **Outcome prediction** — predicting outcomes for a given list of tasks conditioned on a learner's history. Returns the mean of a Beta distribution parameterized by the model's prediction head.
- **Autoregressive forecasting** — sequentially predicting future knowledge states over multiple ordered task sequences in parallel. Uses Beta sampling for outcome feedback during rollout to capture uncertainty.

All endpoints are designed to be called via simple `curl` (or any HTTP client) requests. A Python client (`DFMClient`) is provided for convenience.

## Quickstart

```bash
pip install -e .
dfm-server --checkpoint path/to/ckpt.pt --tasks path/to/tasks.json --device cuda
```

## Design priorities

Fast, memory-efficient KV-cache management is central to the architecture, since every operation relies on maintaining and extending per-learner context. The `BatchedKVCache` forks from a learner's prefix with zero-copy views, allocating only the per-batch suffix for forecast rollouts.
