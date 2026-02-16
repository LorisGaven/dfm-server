"""
DFM model (copied from dfm-training, inference-only) + KV cache for incremental inference.
"""

import copy
import logging
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress noisy HF loading messages (progress bars, load reports)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
import transformers.utils.logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
from transformers import AutoModel, AutoTokenizer  # noqa: E402


@dataclass
class DFMConfig:
    block_size: int = 8192
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_attn_implementation: str = "sdpa"
    # Training-only fields (kept for checkpoint compatibility)
    encoder_grad_batch_size: int = 64
    encoder_nograd_batch_size: int = 4096


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1)


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
    then diverge autoregressively.

    Memory optimization: the shared prefix is stored as a read-only view into the
    source KVCache (zero allocation). Only the per-batch suffix is allocated.
    insert_kv() produces a temporary concatenation per layer call, which the CUDA
    caching allocator reuses across layers.
    """

    def __init__(self, source: KVCache, batch_size: int, extra_len: int):
        prefix_len = source.pos
        n_layers, _, n_kv_head, _, head_dim = source.k.shape
        device, dtype = source.k.device, source.k.dtype

        # View into source prefix (no copy, no allocation)
        if prefix_len > 0:
            self.k_prefix = source.k[:, :, :, :prefix_len, :]
            self.v_prefix = source.v[:, :, :, :prefix_len, :]
        else:
            self.k_prefix = None
            self.v_prefix = None

        # Owned per-batch suffix
        self.k_suffix = torch.zeros(n_layers, batch_size, n_kv_head, extra_len, head_dim, device=device, dtype=dtype)
        self.v_suffix = torch.zeros(n_layers, batch_size, n_kv_head, extra_len, head_dim, device=device, dtype=dtype)

        self.prefix_len = prefix_len
        self.suffix_pos = 0
        self.batch_size = batch_size
        self.max_suffix_len = extra_len

    def insert_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        T_new = k.size(2)
        end = self.suffix_pos + T_new
        assert end <= self.max_suffix_len, (
            f"BatchedKVCache overflow: {self.prefix_len + end} > {self.prefix_len + self.max_suffix_len}"
        )
        self.k_suffix[layer_idx, :, :, self.suffix_pos : end, :] = k
        self.v_suffix[layer_idx, :, :, self.suffix_pos : end, :] = v

        suffix_k = self.k_suffix[layer_idx, :, :, :end, :]
        suffix_v = self.v_suffix[layer_idx, :, :, :end, :]

        if self.k_prefix is not None:
            # expand is a stride-0 view (no copy); cat allocates a temporary
            k_pre = self.k_prefix[layer_idx].expand(self.batch_size, -1, -1, -1)
            v_pre = self.v_prefix[layer_idx].expand(self.batch_size, -1, -1, -1)
            return torch.cat([k_pre, suffix_k], dim=2), torch.cat([v_pre, suffix_v], dim=2)
        return suffix_k, suffix_v

    def get_pos(self) -> int:
        return self.prefix_len + self.suffix_pos

    def advance(self, n: int):
        self.suffix_pos += n

    @classmethod
    def fork(cls, source: "BatchedKVCache", fan_out: int, extra_len: int) -> "BatchedKVCache":
        """Fork each of the B sequences into fan_out copies -> B*fan_out batch.

        Individual i's copies are at batch indices [i*fan_out, ..., i*fan_out+fan_out-1].
        Inherits the prefix view (no copy). Only the suffix is repeat_interleaved.
        """
        B = source.batch_size
        new_B = B * fan_out
        n_layers, _, n_kv_head, _, head_dim = source.k_suffix.shape
        device, dtype = source.k_suffix.device, source.k_suffix.dtype

        obj = object.__new__(cls)

        # Inherit prefix view (same reference, no copy)
        obj.k_prefix = source.k_prefix
        obj.v_prefix = source.v_prefix
        obj.prefix_len = source.prefix_len

        # New suffix: existing content + extra room
        new_suffix_len = source.suffix_pos + extra_len
        obj.k_suffix = torch.zeros(n_layers, new_B, n_kv_head, new_suffix_len, head_dim, device=device, dtype=dtype)
        obj.v_suffix = torch.zeros(n_layers, new_B, n_kv_head, new_suffix_len, head_dim, device=device, dtype=dtype)

        if source.suffix_pos > 0:
            src_k = source.k_suffix[:, :, :, :source.suffix_pos, :]
            src_v = source.v_suffix[:, :, :, :source.suffix_pos, :]
            obj.k_suffix[:, :, :, :source.suffix_pos, :] = src_k.repeat_interleave(fan_out, dim=1)
            obj.v_suffix[:, :, :, :source.suffix_pos, :] = src_v.repeat_interleave(fan_out, dim=1)

        obj.suffix_pos = source.suffix_pos
        obj.batch_size = new_B
        obj.max_suffix_len = new_suffix_len
        return obj


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
        self.task_proj = nn.Linear(config.n_input, config.n_embd, bias=False)
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
    def __init__(self, config, n_input=None):
        """
        Args:
            config: DFMConfig
            n_input: override encoder dim (skips encoder loading, for testing)
        """
        super().__init__()
        self.config = copy.deepcopy(config)
        self.config.block_size = self.config.block_size * 2

        if n_input is not None:
            # Testing mode: skip encoder loading
            self.tokenizer = None
            self.encoder = None
        else:
            # Load encoder and tokenizer
            encoder_name = os.path.expanduser(config.encoder_name)
            explicit_local_path = (
                os.path.isabs(encoder_name)
                or config.encoder_name.startswith(".")
                or config.encoder_name.startswith("~")
            )
            local = explicit_local_path or os.path.isdir(encoder_name)
            if explicit_local_path and not os.path.isdir(encoder_name):
                raise FileNotFoundError(
                    f"encoder_name points to a local path that does not exist: {encoder_name}"
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                encoder_name, local_files_only=local
            )
            self.encoder = AutoModel.from_pretrained(
                encoder_name,
                local_files_only=local,
                attn_implementation=self.config.encoder_attn_implementation,
                dtype=torch.bfloat16,
            )
            n_input = self.encoder.config.hidden_size

        # Store derived n_input on config for EmbeddingLayer
        emb_config = copy.deepcopy(self.config)
        emb_config.n_input = n_input

        self.embedding = EmbeddingLayer(emb_config)
        self.transformer = nn.ModuleDict(
            {
                "h": nn.ModuleList(
                    [Block(self.config, layer_idx) for layer_idx in range(self.config.n_layer)]
                ),
            }
        )
        self.prediction_head = nn.Linear(self.config.n_embd, 2, bias=False)
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

    @torch.no_grad()
    def encode_tasks(self, task_strings: list[str], batch_size: int = 64) -> torch.Tensor:
        """Encode task strings using the built-in sentence encoder.

        Args:
            task_strings: list of task strings to encode
            batch_size: number of strings to encode per batch
        Returns:
            (N, encoder_dim) tensor in model dtype (bfloat16)
        """
        assert self.encoder is not None, "No encoder loaded (test mode?)"
        model_dtype = self.embedding.task_proj.weight.dtype
        all_embs = []
        for i in range(0, len(task_strings), batch_size):
            batch = task_strings[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.get_device())
            attention_mask = encoded["attention_mask"].to(self.get_device())
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embs = _mean_pool(out.last_hidden_state, attention_mask)
            all_embs.append(embs.to(dtype=model_dtype))
        return torch.cat(all_embs, dim=0)

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
        raw = self.prediction_head(valid_task_hiddens)  # (N_valid, 2)
        mean_logit = raw[:, 0].float()
        conc_logit = raw[:, 1].float()
        mean_logit = 15.0 * torch.tanh(mean_logit / 15.0)
        conc_logit = 15.0 * torch.tanh(conc_logit / 15.0)
        mu = torch.sigmoid(mean_logit)
        nu = F.softplus(conc_logit) + 2.0
        alpha = mu * nu
        beta_param = (1.0 - mu) * nu
        # beta-binomial NLL with n_trials=1
        k = valid_outcomes
        n = torch.ones_like(k)
        log_prob = (
            torch.lgamma(k + alpha)
            + torch.lgamma(n - k + beta_param)
            - torch.lgamma(n + alpha + beta_param)
            - torch.lgamma(alpha)
            - torch.lgamma(beta_param)
            + torch.lgamma(alpha + beta_param)
        )
        loss = -log_prob.mean()
        return mean_logit, loss, x.detach()


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
    """Apply prediction head -> mean probability.

    Args:
        hidden: (1, 1, n_embd) — hidden state at a task position
    Returns:
        probability as a Python float
    """
    raw = model.prediction_head(hidden)  # (1, 1, 2)
    raw = raw.float()
    mean_logit = 15.0 * torch.tanh(raw[..., 0] / 15.0)
    return torch.sigmoid(mean_logit).item()


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
    """Apply prediction head -> mean probabilities for a batch.

    Args:
        hidden: (B, 1, n_embd)
    Returns:
        (B,) float tensor of probabilities
    """
    raw = model.prediction_head(hidden).squeeze(-2)  # (B, 2)
    raw = raw.float()
    mean_logit = 15.0 * torch.tanh(raw[:, 0] / 15.0)
    return torch.sigmoid(mean_logit)


def sample_from_hiddens(model: DFM, hidden: torch.Tensor) -> torch.Tensor:
    """Sample outcomes from Beta(alpha, beta) for a batch.

    Used in forecast autoregressive loop — samples from the full Beta distribution
    rather than using the mean, to capture uncertainty in multi-step rollouts.

    Args:
        hidden: (B, 1, n_embd)
    Returns:
        (B,) float tensor of sampled outcomes
    """
    raw = model.prediction_head(hidden).squeeze(-2)  # (B, 2)
    raw = raw.float()
    mean_logit = 15.0 * torch.tanh(raw[:, 0] / 15.0)
    conc_logit = 15.0 * torch.tanh(raw[:, 1] / 15.0)
    mu = torch.sigmoid(mean_logit)
    nu = F.softplus(conc_logit) + 2.0
    alpha = mu * nu
    beta_param = (1.0 - mu) * nu
    return torch.distributions.Beta(alpha, beta_param).sample()  # (B,)
