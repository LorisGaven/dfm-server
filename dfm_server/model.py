"""
DFM model (copied from dfm-training, inference-only) + KV cache for incremental inference.
"""

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DFMConfig:
    block_size: int = 8192
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    n_input: int = 4096


# ---------------------------------------------------------------------------
# KV Cache for incremental inference
# ---------------------------------------------------------------------------


class KVCache:
    """Pre-allocated KV cache for a single sequence (batch=1)."""

    def __init__(self, config: DFMConfig, max_len: int, device: torch.device, dtype: torch.dtype):
        head_dim = config.n_embd // config.n_head
        # Shape: (n_layers, 1, n_kv_head, max_len, head_dim)
        self.k = torch.zeros(config.n_layer, 1, config.n_kv_head, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(config.n_layer, 1, config.n_kv_head, max_len, head_dim, device=device, dtype=dtype)
        self.pos = 0
        self.max_len = max_len

    def insert_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Write new k, v into the cache and return the full view up to current position.

        Args:
            layer_idx: which transformer layer
            k, v: (1, n_kv_head, T_new, head_dim)
        Returns:
            (full_k, full_v) each (1, n_kv_head, pos + T_new, head_dim)
        """
        T_new = k.size(2)
        end = self.pos + T_new
        assert end <= self.max_len, f"KV cache overflow: {end} > {self.max_len}"
        self.k[layer_idx, :, :, self.pos : end, :] = k
        self.v[layer_idx, :, :, self.pos : end, :] = v
        return self.k[layer_idx, :, :, :end, :], self.v[layer_idx, :, :, :end, :]

    def get_pos(self) -> int:
        return self.pos

    def advance(self, n: int):
        self.pos += n


class BatchedKVCache:
    """KV cache forked from a single-sequence prefix into a batch of S sequences.

    Used for batched forecasting: all S sequences share the same prefix,
    then diverge autoregressively. Only allocates prefix_len + extra_len slots.
    """

    def __init__(self, source: KVCache, batch_size: int, extra_len: int):
        prefix_len = source.pos
        total_len = prefix_len + extra_len
        n_layers, _, n_kv_head, _, head_dim = source.k.shape
        device, dtype = source.k.device, source.k.dtype

        self.k = torch.zeros(n_layers, batch_size, n_kv_head, total_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(n_layers, batch_size, n_kv_head, total_len, head_dim, device=device, dtype=dtype)

        # Broadcast-copy the prefix (1 -> batch_size)
        if prefix_len > 0:
            self.k[:, :, :, :prefix_len, :] = source.k[:, :, :, :prefix_len, :]
            self.v[:, :, :, :prefix_len, :] = source.v[:, :, :, :prefix_len, :]

        self.pos = prefix_len
        self.max_len = total_len

    def insert_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        T_new = k.size(2)
        end = self.pos + T_new
        assert end <= self.max_len, f"BatchedKVCache overflow: {end} > {self.max_len}"
        self.k[layer_idx, :, :, self.pos : end, :] = k
        self.v[layer_idx, :, :, self.pos : end, :] = v
        return self.k[layer_idx, :, :, :end, :], self.v[layer_idx, :, :, :end, :]

    def get_pos(self) -> int:
        return self.pos

    def advance(self, n: int):
        self.pos += n


# ---------------------------------------------------------------------------
# Model components (from dfm-training/dfm_training/training/model.py)
# ---------------------------------------------------------------------------


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)
        Tk = k.size(2)

        nrep = self.n_head // self.n_kv_head
        k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)

        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_dim = config.n_embd
        self.task_proj = nn.Sequential(
            nn.Linear(config.n_input, config.n_embd, bias=False),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.n_embd, bias=False),
        )
        self.outcome_proj = nn.Linear(1, config.n_embd, bias=False)
        self.bos_task_emb = nn.Parameter(torch.randn(1, config.n_embd) * 0.02)
        self.bos_outcome_emb = nn.Parameter(torch.randn(1, config.n_embd) * 0.02)
        self.type_emb = nn.Embedding(2, config.n_embd)

    def forward(self, tasks, outcomes):
        B, T, _ = tasks.shape
        D = self.output_dim
        device, dtype = tasks.device, tasks.dtype

        bos_mask = outcomes < 0.0
        task_embs = self.task_proj(tasks)
        task_embs = torch.where(
            bos_mask.unsqueeze(-1),
            self.bos_task_emb.view(1, 1, -1).expand(B, T, -1),
            task_embs,
        )
        outcome_embs = self.outcome_proj(outcomes.unsqueeze(-1).to(dtype=dtype))
        outcome_embs = torch.where(
            bos_mask.unsqueeze(-1),
            self.bos_outcome_emb.view(1, 1, -1).expand(B, T, -1),
            outcome_embs,
        )

        seq_len = 2 * T
        x = torch.empty(B, seq_len, D, device=device, dtype=dtype)
        x[:, 0::2, :] = task_embs
        x[:, 1::2, :] = outcome_embs

        token_types = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        token_types[:, 0::2] = 0
        token_types[:, 1::2] = 1
        x = x + self.type_emb(token_types)

        return x


class DFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.config.block_size = self.config.block_size * 2
        self.embedding = EmbeddingLayer(self.config)
        self.transformer = nn.ModuleDict(
            {
                "h": nn.ModuleList(
                    [Block(self.config, layer_idx) for layer_idx in range(self.config.n_layer)]
                ),
            }
        )
        self.prediction_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rotary_seq_len = self.config.block_size * 10
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        if device is None:
            device = self.embedding.outcome_proj.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def get_device(self):
        return self.embedding.outcome_proj.weight.device

    def forward(self, tasks, outcomes, kv_cache=None):
        B, T, _ = tasks.shape
        x = self.embedding(tasks, outcomes)
        seq_len = x.size(1)

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0 : T0 + seq_len], self.sin[:, T0 : T0 + seq_len]

        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        task_hiddens = x[:, 0::2, :]
        valid = outcomes >= 0
        valid_task_hiddens = task_hiddens[valid]
        valid_outcomes = outcomes[valid]
        logits = self.prediction_head(valid_task_hiddens)
        logits = logits.float()
        logits = 15.0 * torch.tanh(logits / 15.0)
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), valid_outcomes.view(-1))
        return logits, loss, x.detach()


# ---------------------------------------------------------------------------
# Token-level inference helpers
# ---------------------------------------------------------------------------


def embed_task_token(model: DFM, task_emb: torch.Tensor) -> torch.Tensor:
    """Embed a single task token. task_emb: (1, 1, n_input) -> (1, 1, n_embd)."""
    emb = model.embedding.task_proj(task_emb)  # (1, 1, D)
    type_emb = model.embedding.type_emb.weight[0]  # TASK type
    return emb + type_emb


def embed_bos_task_token(model: DFM) -> torch.Tensor:
    """Embed the BOS task token. -> (1, 1, n_embd)."""
    emb = model.embedding.bos_task_emb.unsqueeze(0)  # (1, 1, D)
    type_emb = model.embedding.type_emb.weight[0]
    return emb + type_emb


def embed_outcome_token(model: DFM, outcome: float, is_bos: bool = False) -> torch.Tensor:
    """Embed a single outcome token. -> (1, 1, n_embd)."""
    dtype = model.embedding.outcome_proj.weight.dtype
    device = model.get_device()
    if is_bos:
        emb = model.embedding.bos_outcome_emb.unsqueeze(0)  # (1, 1, D)
    else:
        outcome_t = torch.tensor([[[outcome]]], dtype=dtype, device=device)
        emb = model.embedding.outcome_proj(outcome_t)  # (1, 1, D)
    type_emb = model.embedding.type_emb.weight[1]  # OUTCOME type
    return emb + type_emb


def transformer_forward(model: DFM, x: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
    """Run embedded token(s) through the transformer blocks and advance cache.

    Args:
        x: (1, T, n_embd) — already embedded tokens
        kv_cache: the learner's KV cache
    Returns:
        hidden: (1, T, n_embd)
    """
    T = x.size(1)
    T0 = kv_cache.get_pos()
    cos_sin = model.cos[:, T0 : T0 + T], model.sin[:, T0 : T0 + T]

    x = norm(x)
    for block in model.transformer.h:
        x = block(x, cos_sin, kv_cache)
    x = norm(x)

    kv_cache.advance(T)
    return x


def predict_from_hidden(model: DFM, hidden: torch.Tensor) -> float:
    """Apply prediction head + softcap -> probability.

    Args:
        hidden: (1, 1, n_embd) — hidden state at a task position
    Returns:
        probability as a Python float
    """
    logit = model.prediction_head(hidden)  # (1, 1, 1)
    logit = logit.float()
    logit = 15.0 * torch.tanh(logit / 15.0)
    return torch.sigmoid(logit).item()


# ---------------------------------------------------------------------------
# Batched inference helpers (for multi-sequence forecast)
# ---------------------------------------------------------------------------


def embed_outcome_tokens(model: DFM, outcomes: torch.Tensor) -> torch.Tensor:
    """Embed a batch of outcome tokens.

    Args:
        outcomes: (B,) float tensor of outcome values
    Returns:
        (B, 1, n_embd)
    """
    dtype = model.embedding.outcome_proj.weight.dtype
    x = outcomes.to(dtype=dtype).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    emb = model.embedding.outcome_proj(x)  # (B, 1, D)
    type_emb = model.embedding.type_emb.weight[1]  # OUTCOME type
    return emb + type_emb


def predict_from_hiddens(model: DFM, hidden: torch.Tensor) -> torch.Tensor:
    """Apply prediction head + softcap -> probabilities for a batch.

    Args:
        hidden: (B, 1, n_embd)
    Returns:
        (B,) float tensor of probabilities
    """
    logits = model.prediction_head(hidden).squeeze(-1).squeeze(-1)  # (B,)
    logits = logits.float()
    logits = 15.0 * torch.tanh(logits / 15.0)
    return torch.sigmoid(logits)
