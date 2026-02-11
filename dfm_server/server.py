"""FastAPI inference server for DFM."""

import argparse
import logging
import os
import sys
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Live request log (last N calls in a framed box)
# ---------------------------------------------------------------------------
_req_count = 0
_req_t0 = 0.0
_LOG_SIZE = 5
_req_log: list[tuple[str, str, int, float, float]] = []  # (method, path, status, dt_ms, timestamp)
_frame_drawn = False

# Box inner width (visible characters between │ and │)
_W = 62

# ANSI helpers
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_RST = "\033[0m"


def _pad(visible_len: int) -> str:
    """Return spaces to fill the rest of the box row."""
    return " " * max(0, _W - visible_len)


def _render_log():
    """Redraw the framed log box showing the last N requests."""
    global _frame_drawn

    elapsed = time.time() - _req_t0
    h, rem = divmod(int(elapsed), 3600)
    mn, s = divmod(rem, 60)
    uptime = f"{h}:{mn:02d}:{s:02d}" if h else f"{mn:02d}:{s:02d}"
    n_learners = len(learners)
    count_str = str(_req_count)
    learner_str = str(n_learners)

    # Header: " 3523 reqs │ 1 learners │ uptime 00:41 "
    # Compute visible length manually
    hdr_vis = f" {count_str} reqs  │  {learner_str} learners  │  uptime {uptime} "
    hdr = (
        f" {_BOLD}{count_str}{_RST} reqs  {_DIM}│{_RST}  "
        f"{_BOLD}{learner_str}{_RST} learners  {_DIM}│{_RST}  "
        f"{_DIM}uptime {uptime}{_RST} "
    )
    hdr += _pad(len(hdr_vis))

    # Build content lines
    lines = []
    for method, path, status, dt_ms, ts in _req_log:
        req_elapsed = ts - _req_t0
        rh, rrem = divmod(int(req_elapsed), 3600)
        rmn, rs = divmod(rrem, 60)
        rtime = f"{rh}:{rmn:02d}:{rs:02d}" if rh else f"{rmn:02d}:{rs:02d}"

        sc = _GREEN if status < 400 else _RED
        path_trunc = path[:28]

        # Visible: " 00:41  POST    /predict                      200    5ms "
        vis = f" {rtime}  {method:<7} {path_trunc:<28} {status}  {dt_ms:>5.0f}ms "
        colored = (
            f" {_DIM}{rtime}{_RST}  {_BOLD}{method:<7}{_RST} "
            f"{_CYAN}{path_trunc:<28}{_RST} "
            f"{sc}{status}{_RST}  {_DIM}{dt_ms:>5.0f}ms{_RST} "
        )
        colored += _pad(len(vis))
        lines.append(colored)

    # Pad empty rows
    while len(lines) < _LOG_SIZE:
        empty_vis = " " * _W
        lines.insert(0, empty_vis)

    top = f"  {_DIM}╭{'─' * _W}╮{_RST}"
    mid = f"  {_DIM}│{_RST}{hdr}{_DIM}│{_RST}"
    sep = f"  {_DIM}├{'─' * _W}┤{_RST}"
    bot = f"  {_DIM}╰{'─' * _W}╯{_RST}"

    # Move cursor up to overwrite previous frame
    total_lines = _LOG_SIZE + 4  # top + header + sep + N log lines + bottom
    if _frame_drawn:
        sys.stdout.write(f"\033[{total_lines}A")

    sys.stdout.write(f"\033[K{top}\n")
    sys.stdout.write(f"\033[K{mid}\n")
    sys.stdout.write(f"\033[K{sep}\n")
    for line in lines:
        sys.stdout.write(f"\033[K  {_DIM}│{_RST}{line}{_DIM}│{_RST}\n")
    sys.stdout.write(f"\033[K{bot}\n")
    sys.stdout.flush()
    _frame_drawn = True


def _status_line(method: str, path: str, status: int, dt_ms: float):
    """Record a request and redraw the log box."""
    global _req_count
    _req_count += 1
    _req_log.append((method, path, status, dt_ms, time.time()))
    if len(_req_log) > _LOG_SIZE:
        _req_log.pop(0)
    _render_log()

from .model import (
    BatchedKVCache,
    DFM,
    DFMConfig,
    KVCache,
    embed_bos_task_token,
    embed_outcome_token,
    embed_outcome_tokens,
    embed_task_token,
    predict_from_hidden,
    predict_from_hiddens,
    transformer_forward,
)
from .schemas import (
    DeleteResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RegisterRequest,
    RegisterResponse,
    SearchRequest,
    SearchResponse,
    UpdateRequest,
    UpdateResponse,
)
from .search import run_search

# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------
model: DFM | None = None
task_to_idx: dict[str, int] = {}
emb_tensor: torch.Tensor | None = None  # (n_vocab+1, n_input), row 0 = BOS zeros
learners: dict[str, KVCache] = {}

MAX_SEQ_LEN = 32768  # max interleaved tokens per learner


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def load_model_and_embeddings(checkpoint_path: str, embeddings_path: str, device: str | None = None):
    """Load checkpoint and embeddings."""
    global model, task_to_idx, emb_tensor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with console.status("[bold cyan]Loading embeddings...", spinner="dots") as status:
        t0 = time.time()
        emb_data = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        keys = emb_data["keys"]
        embeddings = emb_data["embeddings"]
        task_to_idx = {k: i + 1 for i, k in enumerate(keys)}
        bos_row = torch.zeros(1, embeddings.shape[1], dtype=embeddings.dtype)
        emb_tensor = torch.cat([bos_row, embeddings], dim=0).to(device=device, dtype=torch.bfloat16)
        t_emb = time.time() - t0
        console.log(f"[green]Loaded {len(keys):,} task embeddings[/] [dim]({t_emb:.1f}s)[/]")

        status.update("[bold cyan]Loading model checkpoint...")
        t0 = time.time()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = DFMConfig(**ckpt["model_args"])
        model = DFM(config)
        state_dict = ckpt["model"]
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.removeprefix("module.").removeprefix("_orig_mod.")] = v
        model.load_state_dict(cleaned)
        t_load = time.time() - t0
        console.log(f"[green]Loaded checkpoint[/] [dim]({t_load:.1f}s)[/]")

        status.update(f"[bold cyan]Moving model to {device}...")
        t0 = time.time()
        model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        t_move = time.time() - t0
        console.log(f"[green]Model ready on {device}[/] [dim]({t_move:.1f}s)[/]")

    n_params = sum(p.numel() for p in model.parameters())
    iter_num = ckpt.get("iter_num", "?")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Parameters", f"{n_params:,}")
    table.add_row("Layers", str(config.n_layer))
    table.add_row("Heads", f"{config.n_head} Q / {config.n_kv_head} KV")
    table.add_row("Embedding dim", str(config.n_embd))
    table.add_row("Input dim", str(config.n_input))
    table.add_row("Vocabulary", f"{len(keys):,} tasks")
    table.add_row("Checkpoint", f"iter {iter_num}")
    table.add_row("Device", str(device))
    console.print(table)


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):
    global _req_t0
    _req_t0 = time.time()
    # Only load real model if env vars are set (allows TestClient to skip)
    if "DFM_CHECKPOINT_PATH" in os.environ:
        load_model_and_embeddings(
            os.environ["DFM_CHECKPOINT_PATH"],
            os.environ["DFM_EMBEDDINGS_PATH"],
            os.environ.get("DFM_DEVICE"),
        )
    yield


app = FastAPI(title="DFM Inference Server", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    dt_ms = (time.time() - t0) * 1000
    _status_line(request.method, request.url.path, response.status_code, dt_ms)
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    return model.get_device()


def get_learner(learner_id: str) -> KVCache:
    if learner_id not in learners:
        raise HTTPException(status_code=404, detail=f"Learner '{learner_id}' not found")
    return learners[learner_id]


def lookup_task_embeddings(tasks: list[str]) -> torch.Tensor:
    """Look up task embeddings by string name. Returns (1, T, n_input)."""
    indices = []
    for t in tasks:
        idx = task_to_idx.get(t)
        if idx is None:
            raise HTTPException(status_code=400, detail=f"Unknown task: '{t}'")
        indices.append(idx)
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=get_device())
    return emb_tensor[idx_tensor].unsqueeze(0)  # (1, T, n_input)


def history_len_from_cache(kv_cache: KVCache) -> int:
    """Number of (task, outcome) pairs in the cache, excluding BOS."""
    # Cache position is in interleaved tokens: 2 for BOS pair + 2 per real pair
    return (kv_cache.get_pos() // 2) - 1


def new_kv_cache() -> KVCache:
    return KVCache(
        model.config,
        max_len=MAX_SEQ_LEN,
        device=get_device(),
        dtype=torch.bfloat16,
    )


@torch.no_grad()
def prefill_bos(kv_cache: KVCache):
    """Prefill a KV cache with the BOS (task, outcome) pair."""
    bos_task = embed_bos_task_token(model)  # (1, 1, D)
    bos_outcome = embed_outcome_token(model, outcome=0.0, is_bos=True)  # (1, 1, D)
    x = torch.cat([bos_task, bos_outcome], dim=1)  # (1, 2, D)
    transformer_forward(model, x, kv_cache)


@torch.no_grad()
def prefill_history(kv_cache: KVCache, task_embs: torch.Tensor, outcomes: list[float]):
    """Prefill with BOS + history using the full model.forward() path.

    This uses model.forward() which does the full EmbeddingLayer interleaving,
    ensuring exact match with training behavior.

    Args:
        task_embs: (1, T, n_input) — task embeddings (BOS zeros already at position 0)
        outcomes: length T — outcomes (-1 for BOS position)
    """
    device = get_device()
    dtype = torch.bfloat16
    outcomes_t = torch.tensor([outcomes], dtype=torch.float32, device=device)
    # Use model.forward() for the embedding + transformer pass
    # This fills the kv_cache as a side effect
    x = model.embedding(task_embs.to(dtype=dtype), outcomes_t)  # (1, 2*T, D)
    seq_len = x.size(1)
    T0 = kv_cache.get_pos()
    cos_sin = model.cos[:, T0 : T0 + seq_len], model.sin[:, T0 : T0 + seq_len]
    x = torch.nn.functional.rms_norm(x, (x.size(-1),))
    for block in model.transformer.h:
        x = block(x, cos_sin, kv_cache)
    kv_cache.advance(seq_len)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/learners", response_model=RegisterResponse)
@torch.no_grad()
def register_learner(req: RegisterRequest):
    if req.learner_id in learners:
        raise HTTPException(status_code=409, detail=f"Learner '{req.learner_id}' already exists")

    kv_cache = new_kv_cache()

    has_history = req.tasks is not None and req.outcomes is not None
    if has_history:
        if len(req.tasks) != len(req.outcomes):
            raise HTTPException(status_code=400, detail="tasks and outcomes must have the same length")
        # Build embedding tensor: BOS zeros + real task embeddings
        bos_emb = torch.zeros(1, 1, emb_tensor.shape[1], device=get_device(), dtype=torch.bfloat16)
        real_embs = lookup_task_embeddings(req.tasks)  # (1, T, n_input)
        all_embs = torch.cat([bos_emb, real_embs], dim=1)  # (1, T+1, n_input)
        all_outcomes = [-1.0] + req.outcomes
        prefill_history(kv_cache, all_embs, all_outcomes)
    else:
        prefill_bos(kv_cache)

    learners[req.learner_id] = kv_cache
    return RegisterResponse(learner_id=req.learner_id, history_len=history_len_from_cache(kv_cache))


@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(req: PredictRequest):
    kv_cache = get_learner(req.learner_id)
    task_embs = lookup_task_embeddings(req.tasks)  # (1, T, n_input)

    saved_pos = kv_cache.get_pos()
    predictions = []

    for i in range(len(req.tasks)):
        single_emb = task_embs[:, i : i + 1, :]  # (1, 1, n_input)
        x = embed_task_token(model, single_emb)
        hidden = transformer_forward(model, x, kv_cache)
        p = predict_from_hidden(model, hidden)
        predictions.append(p)
        # Reset position — stale data beyond saved_pos will be overwritten next time
        kv_cache.pos = saved_pos

    return PredictResponse(learner_id=req.learner_id, predictions=predictions)


@app.post("/update", response_model=UpdateResponse)
@torch.no_grad()
def update(req: UpdateRequest):
    kv_cache = get_learner(req.learner_id)
    task_emb = lookup_task_embeddings([req.task])  # (1, 1, n_input)

    # Feed task token
    x = embed_task_token(model, task_emb)
    transformer_forward(model, x, kv_cache)

    # Feed outcome token
    x = embed_outcome_token(model, req.outcome, is_bos=False)
    transformer_forward(model, x, kv_cache)

    return UpdateResponse(learner_id=req.learner_id, history_len=history_len_from_cache(kv_cache))


@app.post("/forecast", response_model=ForecastResponse)
@torch.no_grad()
def forecast(req: ForecastRequest):
    kv_cache = get_learner(req.learner_id)
    S = len(req.task_sequences)
    if S == 0:
        return ForecastResponse(learner_id=req.learner_id, predictions=[])
    L = len(req.task_sequences[0])
    if any(len(seq) != L for seq in req.task_sequences):
        raise HTTPException(status_code=400, detail="All task sequences must have the same length")
    if L == 0:
        return ForecastResponse(learner_id=req.learner_id, predictions=[[] for _ in range(S)])

    # Look up all embeddings: (S, L, n_input)
    # Flatten all unique tasks, look up, then reshape
    all_embs = []
    for seq in req.task_sequences:
        embs = lookup_task_embeddings(seq)  # (1, L, n_input)
        all_embs.append(embs)
    task_embs = torch.cat(all_embs, dim=0)  # (S, L, n_input)

    # Fork KV cache: copies prefix, allocates space for 2*L new tokens (task+outcome per step)
    batched_cache = BatchedKVCache(kv_cache, batch_size=S, extra_len=2 * L)

    predictions = [[] for _ in range(S)]

    for t in range(L):
        # Task embeddings for step t: (S, 1, n_input)
        step_embs = task_embs[:, t : t + 1, :]

        # Embed task tokens → transformer forward → predict
        x = embed_task_token(model, step_embs)  # (S, 1, D)
        hidden = transformer_forward(model, x, batched_cache)  # (S, 1, D)
        probs = predict_from_hiddens(model, hidden)  # (S,)

        for s in range(S):
            predictions[s].append(probs[s].item())

        # Embed predicted outcomes → transformer forward (advance cache for next step)
        x = embed_outcome_tokens(model, probs)  # (S, 1, D)
        transformer_forward(model, x, batched_cache)

    # Original learner cache is untouched (we worked on a fork)
    return ForecastResponse(learner_id=req.learner_id, predictions=predictions)


@app.post("/search", response_model=SearchResponse)
@torch.no_grad()
def search(req: SearchRequest):
    kv_cache = get_learner(req.learner_id)

    if req.depth <= 0:
        raise HTTPException(status_code=400, detail="depth must be > 0")
    if not req.target_tasks:
        raise HTTPException(status_code=400, detail="target_tasks must not be empty")
    if not req.candidate_tasks:
        raise HTTPException(status_code=400, detail="candidate_tasks must not be empty")
    if req.population_size < req.elite_count:
        raise HTTPException(status_code=400, detail="population_size must be >= elite_count")
    if kv_cache.pos + 2 * req.depth > MAX_SEQ_LEN:
        raise HTTPException(status_code=400, detail="Search depth would exceed MAX_SEQ_LEN")

    # Validate and look up target task indices
    target_indices = []
    for t in req.target_tasks:
        idx = task_to_idx.get(t)
        if idx is None:
            raise HTTPException(status_code=400, detail=f"Unknown target task: '{t}'")
        target_indices.append(idx)

    # Validate and look up candidate task indices
    candidate_indices = []
    candidate_names = []
    for t in req.candidate_tasks:
        idx = task_to_idx.get(t)
        if idx is None:
            raise HTTPException(status_code=400, detail=f"Unknown candidate task: '{t}'")
        candidate_indices.append(idx)
        candidate_names.append(t)

    best_sequence, best_fitness = run_search(
        model=model,
        kv_cache=kv_cache,
        candidate_indices=candidate_indices,
        target_indices=target_indices,
        candidate_names=candidate_names,
        emb_tensor=emb_tensor,
        depth=req.depth,
        population_size=req.population_size,
        generations=req.generations,
        elite_count=req.elite_count,
        tournament_size=req.tournament_size,
        crossover_rate=req.crossover_rate,
        mutation_rate=req.mutation_rate,
        eval_every=req.eval_every,
        seed=req.seed,
    )

    return SearchResponse(
        learner_id=req.learner_id,
        best_sequence=best_sequence,
        best_fitness=best_fitness,
    )


@app.delete("/learners/{learner_id}", response_model=DeleteResponse)
def delete_learner(learner_id: str):
    if learner_id not in learners:
        raise HTTPException(status_code=404, detail=f"Learner '{learner_id}' not found")
    del learners[learner_id]
    return DeleteResponse(learner_id=learner_id)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", learner_count=len(learners))


@app.get("/tasks")
def list_tasks():
    return {"tasks": sorted(task_to_idx.keys()), "count": len(task_to_idx)}


@app.get("/config")
def get_config():
    cfg = model.config
    return {
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_kv_head": cfg.n_kv_head,
        "n_embd": cfg.n_embd,
        "n_input": cfg.n_input,
        "block_size": cfg.block_size,
    }


@app.post("/gc")
def gc_cuda():
    """Free cached CUDA memory. Useful after OOM recovery."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


LOGO = r"""
  [bold white]██████╗ ███████╗███╗   ███╗[/]
  [bold white]██╔══██╗██╔════╝████╗ ████║[/]
  [bold white]██║  ██║█████╗  ██╔████╔██║[/]  [bold white] ❀ Developmental Foundation Model[/]
  [bold white]██║  ██║██╔══╝  ██║╚██╔╝██║[/]     [dim]── Inference Server v0.1.0 ──[/]
  [bold white]██████╔╝██║     ██║ ╚═╝ ██║[/]
  [bold white]╚═════╝ ╚═╝     ╚═╝     ╚═╝[/]
"""


def main():
    parser = argparse.ArgumentParser(description="DFM Inference Server")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings file (.pt)")
    parser.add_argument("--device", default=None, help="Device (default: cuda if available)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global _req_t0
    _req_t0 = time.time()

    console.print(LOGO)
    load_model_and_embeddings(args.checkpoint, args.embeddings, args.device)
    console.print()
    console.rule("[bold green]Serving")
    console.print(f"  Listening on [bold]http://{args.host}:{args.port}[/]")
    console.print(f"  Endpoints:   [cyan]/learners  /predict  /update  /forecast  /search  /tasks  /health[/]")
    console.print()

    # Suppress uvicorn's default access/info logs — we have our own status line
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
