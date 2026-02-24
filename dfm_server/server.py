"""FastAPI inference server for DFM."""

import argparse
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from rich.console import Console
from rich.table import Table

from .model import (
    BatchedKVCache,
    DFM,
    DFMConfig,
    KVCache,
    predict_from_hiddens,
    transformer_forward,
)
from .schemas import (
    DeleteResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RegisterRequest,
    RegisterResponse,
    UpdateRequest,
    UpdateResponse,
)

console = Console()

# ---------------------------------------------------------------------------
# Live request log (last N calls in a framed box)
# ---------------------------------------------------------------------------
_req_count = 0
_req_t0 = 0.0
_LOG_SIZE = 5
_req_log: list[
    tuple[str, str, int, float, float]
] = []  # (method, path, status, dt_ms, timestamp)
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


# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------
model: DFM | None = None
task_to_idx: dict[str, int] = {}
task_emb_table: torch.Tensor | None = (
    None  # (V+1, n_embd) pre-projected through task_proj
)
max_context_size: int = 512
encode_batch_size: int = 64

MAX_SEQ_LEN = 32768  # max tokens per learner (context + curriculum + targets)


@dataclass
class LearnerState:
    """Stores the last N (task, outcome, answer) entries for a learner."""

    history: list[tuple[int, float, torch.Tensor | None]] = field(default_factory=list)
    # Each entry: (task_idx, outcome, pre-projected_answer_emb or None)
    # answer_emb is (n_embd,) already projected through answer_proj, or None if "[None]"


learners: dict[str, LearnerState] = {}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str,
    tasks_path: str,
    device: str | None = None,
    max_context_size_: int = 512,
    encode_batch_size_: int = 64,
):
    """Load checkpoint with trained encoder, tasks, and pre-encode task embeddings."""
    global model, task_to_idx, task_emb_table, max_context_size, encode_batch_size

    max_context_size = max_context_size_
    encode_batch_size = encode_batch_size_

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with console.status(
        "[bold cyan]Loading model checkpoint...", spinner="dots"
    ) as status:
        t0 = time.time()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = DFMConfig(**ckpt["model_args"])
        model = DFM(config)

        # Create TextEncoder so encoder.* keys exist when loading state dict
        enc_cfg = ckpt["encoder_config"]
        # Suppress noisy logs and tqdm progress bars from transformers
        for name in ("transformers", "huggingface_hub"):
            logging.getLogger(name).setLevel(logging.WARNING)
        with open(os.devnull, "w") as devnull:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            try:
                model.load_encoder(enc_cfg["model_name"], enc_cfg["max_length"])
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

        # Load full state dict (includes trained encoder.model.* weights)
        state_dict = ckpt["model"]
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.removeprefix("module.").removeprefix("_orig_mod.")] = v
        result = model.load_state_dict(cleaned, strict=False)
        if result.missing_keys:
            console.log(
                f"[yellow]Missing keys:[/] {result.missing_keys}"
            )
        unexpected = [
            k for k in result.unexpected_keys
            if not k.startswith(("task_embeddings", "answer_embeddings",
                                 "task_token_ids", "task_attention_masks",
                                 "answer_token_ids", "answer_attention_masks",
                                 "task_embedding_cache", "answer_embedding_cache"))
        ]
        if unexpected:
            console.log(f"[yellow]Unexpected keys:[/] {unexpected}")
        t_load = time.time() - t0
        console.log(f"[green]Loaded checkpoint[/] [dim]({t_load:.1f}s)[/]")

        status.update(f"[bold cyan]Moving model to {device}...")
        t0 = time.time()
        model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        t_move = time.time() - t0
        console.log(f"[green]Model ready on {device}[/] [dim]({t_move:.1f}s)[/]")

        # Load tasks
        status.update("[bold cyan]Loading tasks...")
        t0 = time.time()
        with open(tasks_path) as f:
            tasks_list = json.load(f)
        task_to_idx = {t: i + 1 for i, t in enumerate(tasks_list)}
        t_tasks = time.time() - t0
        console.log(
            f"[green]Loaded {len(tasks_list):,} tasks[/] [dim]({t_tasks:.1f}s)[/]"
        )

        # Encode all tasks with trained encoder
        status.update("[bold cyan]Encoding tasks...")
        t0 = time.time()
        raw_task_embs = _encode_strings(tasks_list)  # (N, n_input)
        t_encode = time.time() - t0
        console.log(
            f"[green]Encoded {len(tasks_list):,} tasks[/] [dim]({t_encode:.1f}s)[/]"
        )

        # Pre-project through task_proj
        status.update("[bold cyan]Pre-projecting task embeddings...")
        t0 = time.time()
        n_input = raw_task_embs.shape[1]
        bos_row = torch.zeros(
            1, n_input, dtype=raw_task_embs.dtype, device=raw_task_embs.device
        )
        all_raw = torch.cat([bos_row, raw_task_embs], dim=0).to(
            device=device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            task_emb_table = model.embedding.task_proj(all_raw)  # (N+1, n_embd)
        t_proj = time.time() - t0
        console.log(f"[green]Pre-projected embeddings[/] [dim]({t_proj:.1f}s)[/]")

    n_params = sum(p.numel() for p in model.parameters())

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Parameters", f"{n_params:,}")
    table.add_row("Layers", str(config.n_layer))
    table.add_row("Heads", f"{config.n_head} Q / {config.n_kv_head} KV")
    table.add_row("Embedding dim", str(config.n_embd))
    table.add_row("Input dim", str(n_input))
    table.add_row("Encoder", enc_cfg["model_name"])
    table.add_row("Vocabulary", f"{len(tasks_list):,} tasks")
    table.add_row("Max context", str(max_context_size))
    table.add_row("Device", str(device))
    console.print(table)


@asynccontextmanager
async def lifespan(app):
    global _req_t0
    _req_t0 = time.time()
    # Only load real model if env vars are set (allows TestClient to skip)
    if "DFM_CHECKPOINT_PATH" in os.environ:
        load_model(
            os.environ["DFM_CHECKPOINT_PATH"],
            os.environ["DFM_TASKS_PATH"],
            os.environ.get("DFM_DEVICE"),
            int(os.environ.get("DFM_MAX_CONTEXT_SIZE", "512")),
            int(os.environ.get("DFM_ENCODE_BATCH_SIZE", "64")),
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


def get_learner(learner_id: str) -> LearnerState:
    if learner_id not in learners:
        raise HTTPException(status_code=404, detail=f"Learner '{learner_id}' not found")
    return learners[learner_id]


def lookup_task_idx(task: str) -> int:
    """Look up a task string's index. Raises 400 if unknown."""
    idx = task_to_idx.get(task)
    if idx is None:
        raise HTTPException(status_code=400, detail=f"Unknown task: '{task}'")
    return idx


@torch.no_grad()
def _encode_strings(strings: list[str]) -> torch.Tensor:
    """Encode strings using the model's trained encoder.

    Returns (N, n_input) tensor on the model's device in bfloat16.
    """
    device = get_device()
    all_embs = []
    for i in range(0, len(strings), encode_batch_size):
        batch = strings[i : i + encode_batch_size]
        tokens = model.encoder.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=model.encoder.max_length,
            return_tensors="pt",
        ).to(device)
        embs = model.encoder(
            tokens["input_ids"].long(), tokens["attention_mask"].long()
        )
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


def encode_answers(answer_strs: list[str]) -> list[torch.Tensor | None]:
    """Encode answer strings in batch and project through answer_proj.

    Returns list of (n_embd,) tensors, or None for "[None]" entries.
    """
    # Find non-None answers and their positions
    to_encode = []
    indices = []
    for i, s in enumerate(answer_strs):
        if s != "[None]":
            to_encode.append(s)
            indices.append(i)

    results: list[torch.Tensor | None] = [None] * len(answer_strs)

    if to_encode:
        raw_embs = _encode_strings(to_encode)  # (M, n_input)
        with torch.no_grad():
            projected = model.embedding.answer_proj(raw_embs)  # (M, n_embd)
        for j, idx in enumerate(indices):
            results[idx] = projected[j]

    return results


@torch.no_grad()
def build_context_and_prefill(learner: LearnerState) -> KVCache:
    """Build context token sequence from learner history and prefill a KV cache.

    Returns a fresh KVCache with the context prefilled.
    """
    device = get_device()
    dtype = torch.bfloat16
    D_embd = model.config.n_embd
    # Count tokens: 1 BOS + per-entry (2 or 3 tokens)
    n_tokens = 1
    for _, _, answer_emb in learner.history:
        n_tokens += 3 if answer_emb is not None else 2

    # Allocate token type array
    token_types = torch.zeros(1, n_tokens, dtype=torch.long, device=device)

    # Position 0: BOS token (type=0 already from zeros init)
    pos = 1
    for _, _, answer_emb in learner.history:
        token_types[0, pos] = 1  # TASK
        pos += 1
        token_types[0, pos] = 2  # OUTCOME
        pos += 1
        if answer_emb is not None:
            token_types[0, pos] = 3  # ANSWER
            pos += 1

    # Since we have pre-projected task and answer embeddings, we need to build
    # the embedded sequence manually (bypassing task_proj and answer_proj).
    x = torch.zeros(1, n_tokens, D_embd, dtype=dtype, device=device)

    # BOS: learned embedding
    x[0, 0] = model.embedding.bos_emb.to(dtype)

    pos = 1
    for task_idx, outcome, answer_emb in learner.history:
        # TASK: pre-projected task embedding from table
        x[0, pos] = task_emb_table[task_idx]
        pos += 1

        # OUTCOME: project through outcome_proj
        outcome_t = torch.tensor([[[outcome]]], dtype=dtype, device=device)
        x[0, pos] = model.embedding.outcome_proj(outcome_t).squeeze()
        pos += 1

        # ANSWER: pre-projected answer embedding
        if answer_emb is not None:
            x[0, pos] = answer_emb
            pos += 1

    # Add type embeddings
    x = x + model.embedding.type_emb(token_types).to(dtype)

    # Create and prefill KV cache
    kv_cache = KVCache(
        model.config,
        max_len=MAX_SEQ_LEN,
        device=device,
        dtype=dtype,
    )
    transformer_forward(model, x, kv_cache)
    return kv_cache


@torch.no_grad()
def build_task_tokens(task_indices: list[int]) -> torch.Tensor:
    """Build embedded TASK tokens from pre-projected task embedding table.

    Args:
        task_indices: list of task indices
    Returns:
        (1, T, n_embd) embedded tokens
    """
    device = get_device()
    dtype = torch.bfloat16
    T = len(task_indices)
    D_embd = model.config.n_embd

    x = torch.zeros(1, T, D_embd, dtype=dtype, device=device)
    idx_tensor = torch.tensor(task_indices, dtype=torch.long, device=device)
    x[0] = task_emb_table[idx_tensor]

    # Add TASK type embedding (type=1)
    type_emb = model.embedding.type_emb.weight[1].to(dtype)
    x = x + type_emb

    return x


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/learners", response_model=RegisterResponse)
@torch.no_grad()
def register_learner(req: RegisterRequest):
    if req.learner_id in learners:
        raise HTTPException(
            status_code=409, detail=f"Learner '{req.learner_id}' already exists"
        )

    state = LearnerState()

    has_history = req.tasks is not None and req.outcomes is not None
    if has_history:
        if len(req.tasks) != len(req.outcomes):
            raise HTTPException(
                status_code=400, detail="tasks and outcomes must have the same length"
            )
        answers = req.answers or ["[None]"] * len(req.tasks)
        if len(answers) != len(req.tasks):
            raise HTTPException(
                status_code=400, detail="answers must have the same length as tasks"
            )

        answer_embs = encode_answers(answers)
        for task_str, outcome, answer_emb in zip(req.tasks, req.outcomes, answer_embs):
            task_idx = lookup_task_idx(task_str)
            state.history.append((task_idx, outcome, answer_emb))

        # Truncate to max context size
        if len(state.history) > max_context_size:
            state.history = state.history[-max_context_size:]

    learners[req.learner_id] = state
    return RegisterResponse(learner_id=req.learner_id, history_len=len(state.history))


@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(req: PredictRequest):
    learner = get_learner(req.learner_id)

    has_curriculum = req.curriculum is not None
    has_targets = req.target_tasks is not None

    if not has_curriculum and not has_targets:
        raise HTTPException(
            status_code=400,
            detail="At least one of curriculum or target_tasks must be provided",
        )

    # Validate curriculum
    S = 1  # default: single "curriculum" (no curriculum case)
    L = 0
    if has_curriculum:
        S = len(req.curriculum)
        if S == 0:
            raise HTTPException(status_code=400, detail="curriculum must not be empty")
        L = len(req.curriculum[0])
        if any(len(seq) != L for seq in req.curriculum):
            raise HTTPException(
                status_code=400,
                detail="All curriculum sequences must have the same length",
            )

    # Build context and prefill KV cache (batch=1)
    kv_cache = build_context_and_prefill(learner)

    if has_targets:
        T = len(req.target_tasks)
        target_indices = [lookup_task_idx(t) for t in req.target_tasks]

        if has_curriculum and L > 0:
            # Fork context to S for curriculum
            batched_cache = BatchedKVCache(kv_cache, batch_size=S, extra_len=L)

            # Build and forward curriculum tokens: (S, L, n_embd)
            curriculum_x_list = []
            for seq in req.curriculum:
                indices = [lookup_task_idx(t) for t in seq]
                curriculum_x_list.append(build_task_tokens(indices))
            curriculum_x = torch.cat(curriculum_x_list, dim=0)  # (S, L, n_embd)
            transformer_forward(model, curriculum_x, batched_cache)

            # Fork S -> S*T so each target is independent per curriculum
            target_cache = BatchedKVCache.fork(batched_cache, fan_out=T, extra_len=1)
        else:
            # No curriculum — fork context directly to S*T
            # S=1 when no curriculum, so this gives T caches
            target_cache = BatchedKVCache(kv_cache, batch_size=S * T, extra_len=1)

        # Build target tokens: one token each, (S*T, 1, n_embd)
        # Layout per curriculum s: [s_t0, s_t1, ..., s_tT-1]
        target_x = build_task_tokens(target_indices)  # (1, T, n_embd)
        target_x = target_x.expand(S, -1, -1).reshape(S * T, 1, -1)

        hidden = transformer_forward(model, target_x, target_cache)  # (S*T, 1, n_embd)
        preds = predict_from_hiddens(model, hidden).reshape(S, T)  # (S, T)

        predictions = preds.tolist()
    else:
        # Curriculum only, predict at each curriculum position
        # Fork context to S
        batched_cache = BatchedKVCache(kv_cache, batch_size=S, extra_len=L)

        # Build and forward curriculum tokens: (S, L, n_embd)
        curriculum_x_list = []
        for seq in req.curriculum:
            indices = [lookup_task_idx(t) for t in seq]
            curriculum_x_list.append(build_task_tokens(indices))
        curriculum_x = torch.cat(curriculum_x_list, dim=0)  # (S, L, n_embd)

        hidden = transformer_forward(
            model, curriculum_x, batched_cache
        )  # (S, L, n_embd)
        preds = predict_from_hiddens(model, hidden)  # (S, L)

        predictions = preds.tolist()

    return PredictResponse(learner_id=req.learner_id, predictions=predictions)


@app.post("/update", response_model=UpdateResponse)
@torch.no_grad()
def update(req: UpdateRequest):
    learner = get_learner(req.learner_id)

    if len(req.tasks) != len(req.outcomes):
        raise HTTPException(
            status_code=400, detail="tasks and outcomes must have the same length"
        )
    answers = req.answers or ["[None]"] * len(req.tasks)
    if len(answers) != len(req.tasks):
        raise HTTPException(
            status_code=400, detail="answers must have the same length as tasks"
        )

    answer_embs = encode_answers(answers)
    for task_str, outcome, answer_emb in zip(req.tasks, req.outcomes, answer_embs):
        task_idx = lookup_task_idx(task_str)
        learner.history.append((task_idx, outcome, answer_emb))

    # Truncate to max context size
    if len(learner.history) > max_context_size:
        learner.history = learner.history[-max_context_size:]

    return UpdateResponse(learner_id=req.learner_id, history_len=len(learner.history))


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
        "max_context_size": max_context_size,
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
  [bold white]██║  ██║██╔══╝  ██║╚██╔╝██║[/]     [dim]── Inference Server v0.2.0 ──[/]
  [bold white]██████╔╝██║     ██║ ╚═╝ ██║[/]
  [bold white]╚═════╝ ╚═╝     ╚═╝     ╚═╝[/]
"""


def main():
    parser = argparse.ArgumentParser(description="DFM Inference Server")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument("--tasks", required=True, help="Path to tasks list (.json)")
    parser.add_argument(
        "--max-context-size",
        type=int,
        default=8192,
        help="Max history entries per learner",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
        help="Batch size for encoding tasks at startup",
    )
    parser.add_argument(
        "--device", default=None, help="Device (default: cuda if available)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global _req_t0
    _req_t0 = time.time()

    console.print(LOGO)
    load_model(
        args.checkpoint,
        args.tasks,
        args.device,
        args.max_context_size,
        args.encode_batch_size,
    )
    console.print()
    console.rule("[bold green]Serving")
    console.print(f"  Listening on [bold]http://{args.host}:{args.port}[/]")
    console.print(
        "  Endpoints:   [cyan]/learners  /predict  /update  /tasks  /health[/]"
    )
    console.print()

    # Suppress uvicorn's default access/info logs — we have our own status line
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
