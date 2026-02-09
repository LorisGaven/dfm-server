# dfm-server

Inference server for models trained with [dfm-training](https://github.com/…/dfm-training). The server exposes a set of HTTP endpoints that support a range of downstream applications — benchmark performance prediction, curriculum design, hyperparameter optimization, and more.

## Core capabilities

The API supports operations such as:

- **Registering learners** — either with a prefilled history of tasks/outcomes or from scratch (BOS-only).
- **Outcome prediction** — predicting outcomes for a given list of tasks conditioned on a learner's history.
- **Autoregressive forecasting** — sequentially predicting future knowledge states over an ordered sequence of upcoming tasks.

All endpoints are designed to be called via simple `curl` (or any HTTP client) requests.

## Design priorities

Fast, memory-efficient KV-cache management is central to the architecture, since every operation relies on maintaining and extending per-learner context.