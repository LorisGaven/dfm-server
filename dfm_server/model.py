"""
DFM model (synced from dfm-training, inference-only) + KV cache for incremental inference.

Architecture: 4 token types (BOS=0, TASK=1, OUTCOME=2, ANSWER=3),
BCE prediction head (single logit per task position).
"""

import copy
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


class TextEncoder(nn.Module):
    """Trainable text encoder for task/answer string encoding.

    Loads a pre-trained model (e.g. sentence-transformers/all-MiniLM-L6-v2)
    and fine-tunes all parameters. Outputs mean-pooled hidden states.
    """

    def __init__(self, model_name, max_length=512):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.bfloat16)
        self.hidden_size = self.model.config.hidden_size
        # Disable pooler (e.g. BERT's [CLS] pooler) — we use mean pooling,
        # and unused params break DDP (no gradients → reduction error)
        if hasattr(self.model, "pooler"):
            self.model.pooler = None
        print(
            f"TextEncoder: {model_name}, "
            f"hidden_size={self.hidden_size}, max_length={max_length}"
        )

    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def pretokenize(self, strings, device="cpu"):
        """Tokenize a list of strings. Returns (token_ids, attention_masks).

        Args:
            strings: list of strings where index 0 is "" (BOS placeholder)
            device: device to place tensors on
        Returns:
            token_ids: (N, max_length) int32
            attention_masks: (N, max_length) int8
        """
        import numpy as np

        real_strings = strings[1:]  # skip BOS placeholder
        N = len(real_strings)
        max_len = self.max_length

        # Pre-allocate numpy arrays (row 0 = BOS, zeros)
        np_ids = np.zeros((N + 1, max_len), dtype=np.int32)
        np_masks = np.zeros((N + 1, max_len), dtype=np.int8)

        if N > 0:
            # Tokenize without padding — fast Rust path, no tensor overhead
            batch_size = 100000
            for i in range(0, N, batch_size):
                batch = real_strings[i : i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=False,
                    truncation=True,
                    max_length=max_len,
                    return_attention_mask=False,
                )
                for j, ids in enumerate(encoded["input_ids"]):
                    L = len(ids)
                    np_ids[i + 1 + j, :L] = ids
                    np_masks[i + 1 + j, :L] = 1
                print(
                    f"  Tokenized {min(i + batch_size, N)}/{N} strings",
                    flush=True,
                )

        # torch.from_numpy is zero-copy
        token_ids = torch.from_numpy(np_ids).to(device)
        attention_masks = torch.from_numpy(np_masks).to(device)

        return token_ids, attention_masks


# ---------------------------------------------------------------------------
# KV Cache for incremental inference
# ---------------------------------------------------------------------------


class KVCache:
    """Pre-allocated KV cache for a single sequence (batch=1)."""

    def __init__(
        self, config: DFMConfig, max_len: int, device: torch.device, dtype: torch.dtype
    ):
        head_dim = config.n_embd // config.n_head
        # Shape: (n_layers, 1, n_kv_head, max_len, head_dim)
        self.k = torch.zeros(
            config.n_layer,
            1,
            config.n_kv_head,
            max_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.v = torch.zeros(
            config.n_layer,
            1,
            config.n_kv_head,
            max_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
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

    Three-level memory hierarchy for efficient forking:
      - prefix: shared across all batches, view into source KVCache (batch=1, zero-copy)
      - mid: per-source-batch frozen KV, view into parent's suffix (batch=S, zero-copy).
             Expanded lazily to batch=S*T via repeat_interleave per layer call (temporary).
      - suffix: per-batch writable KV (batch=S or S*T, allocated)

    This avoids permanently copying S*L entries into S*T*L on fork. Instead, fork()
    keeps a view reference to the source's suffix (mid) and only allocates the new
    suffix for extra_len tokens. The repeat_interleave happens as a temporary per
    layer in insert_kv(), reused by the CUDA caching allocator.
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

        # No mid level when created directly from KVCache
        self.k_mid = None
        self.v_mid = None
        self.mid_len = 0
        self.mid_fan_out = 1

        # Owned per-batch suffix
        self.k_suffix = torch.zeros(
            n_layers,
            batch_size,
            n_kv_head,
            extra_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_suffix = torch.zeros(
            n_layers,
            batch_size,
            n_kv_head,
            extra_len,
            head_dim,
            device=device,
            dtype=dtype,
        )

        self.prefix_len = prefix_len
        self.suffix_pos = 0
        self.batch_size = batch_size
        self.max_suffix_len = extra_len

    def insert_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        T_new = k.size(2)
        end = self.suffix_pos + T_new
        assert end <= self.max_suffix_len, (
            f"BatchedKVCache overflow: {self.prefix_len + self.mid_len + end} "
            f"> {self.prefix_len + self.mid_len + self.max_suffix_len}"
        )
        self.k_suffix[layer_idx, :, :, self.suffix_pos : end, :] = k
        self.v_suffix[layer_idx, :, :, self.suffix_pos : end, :] = v

        # Collect parts to concatenate
        k_parts = []
        v_parts = []

        if self.k_prefix is not None:
            # expand is a stride-0 view (no copy)
            k_parts.append(self.k_prefix[layer_idx].expand(self.batch_size, -1, -1, -1))
            v_parts.append(self.v_prefix[layer_idx].expand(self.batch_size, -1, -1, -1))

        if self.k_mid is not None:
            # Lazy expansion: repeat_interleave S -> S*fan_out (temporary per layer)
            k_parts.append(
                self.k_mid[layer_idx].repeat_interleave(self.mid_fan_out, dim=0)
            )
            v_parts.append(
                self.v_mid[layer_idx].repeat_interleave(self.mid_fan_out, dim=0)
            )

        k_parts.append(self.k_suffix[layer_idx, :, :, :end, :])
        v_parts.append(self.v_suffix[layer_idx, :, :, :end, :])

        return torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2)

    def get_pos(self) -> int:
        return self.prefix_len + self.mid_len + self.suffix_pos

    def advance(self, n: int):
        self.suffix_pos += n

    @classmethod
    def fork(
        cls, source: "BatchedKVCache", fan_out: int, extra_len: int
    ) -> "BatchedKVCache":
        """Fork each of the B sequences into fan_out copies -> B*fan_out batch.

        Individual i's copies are at batch indices [i*fan_out, ..., i*fan_out+fan_out-1].

        Memory-efficient: inherits prefix view (no copy), keeps a view reference to
        the source's suffix as mid (no copy), and only allocates a new suffix for
        extra_len tokens. The mid is expanded lazily via repeat_interleave in insert_kv().
        """
        new_B = source.batch_size * fan_out
        n_layers, _, n_kv_head, _, head_dim = source.k_suffix.shape
        device, dtype = source.k_suffix.device, source.k_suffix.dtype

        obj = object.__new__(cls)

        # Inherit prefix view (same reference, no copy)
        obj.k_prefix = source.k_prefix
        obj.v_prefix = source.v_prefix
        obj.prefix_len = source.prefix_len

        # Merge source's mid + suffix into the new mid (both are views, no copy)
        # If source has a mid, we need to materialize mid+suffix into one buffer
        # to avoid recursive nesting. For our use case (single fork level), source
        # never has a mid, so we just reference the source's suffix.
        if source.k_mid is not None:
            # Source already has a mid — materialize mid+suffix into a single tensor
            total_mid_len = source.mid_len + source.suffix_pos
            if total_mid_len > 0:
                # We must copy here to merge the two levels
                obj.k_mid = torch.cat(
                    [
                        source.k_mid[:, :, :, : source.mid_len, :].repeat_interleave(
                            source.mid_fan_out, dim=1
                        ),
                        source.k_suffix[:, :, :, : source.suffix_pos, :],
                    ],
                    dim=3,
                )
                obj.v_mid = torch.cat(
                    [
                        source.v_mid[:, :, :, : source.mid_len, :].repeat_interleave(
                            source.mid_fan_out, dim=1
                        ),
                        source.v_suffix[:, :, :, : source.suffix_pos, :],
                    ],
                    dim=3,
                )
                obj.mid_len = total_mid_len
            else:
                obj.k_mid = None
                obj.v_mid = None
                obj.mid_len = 0
        elif source.suffix_pos > 0:
            # View into source's suffix (no copy)
            obj.k_mid = source.k_suffix[:, :, :, : source.suffix_pos, :]
            obj.v_mid = source.v_suffix[:, :, :, : source.suffix_pos, :]
            obj.mid_len = source.suffix_pos
        else:
            obj.k_mid = None
            obj.v_mid = None
            obj.mid_len = 0

        obj.mid_fan_out = fan_out

        # Only allocate suffix for extra_len tokens (the new writable part)
        obj.k_suffix = torch.zeros(
            n_layers,
            new_B,
            n_kv_head,
            extra_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        obj.v_suffix = torch.zeros(
            n_layers,
            new_B,
            n_kv_head,
            extra_len,
            head_dim,
            device=device,
            dtype=dtype,
        )

        obj.suffix_pos = 0
        obj.batch_size = new_B
        obj.max_suffix_len = extra_len
        return obj


# ---------------------------------------------------------------------------
# Model components (synced from dfm-training/dfm_training/training/model.py)
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
    """Projects tasks, outcomes, and answers to the same embedding space."""

    def __init__(self, config):
        super().__init__()
        self.output_dim = config.n_embd
        self.task_proj = nn.Linear(config.n_input, config.n_embd, bias=False)
        self.outcome_proj = nn.Linear(1, config.n_embd, bias=False)
        self.answer_proj = nn.Linear(config.n_input, config.n_embd, bias=False)
        self.bos_emb = nn.Parameter(torch.randn(config.n_embd) * 0.02)

        # Token type embeddings: 0=BOS, 1=TASK, 2=OUTCOME, 3=ANSWER
        self.type_emb = nn.Embedding(4, config.n_embd)

    def forward(self, token_types, task_embs, answer_embs, outcome_values):
        """
        Args:
            token_types: (B, T) int - 0=BOS, 1=TASK, 2=OUTCOME, 3=ANSWER
            task_embs: (B, T, n_input) - pre-computed task embeddings (zeros at non-TASK)
            answer_embs: (B, T, n_input) - pre-computed answer embeddings (zeros at non-ANSWER)
            outcome_values: (B, T) float - outcome at OUTCOME positions, -1 at BOS, 0 elsewhere

        Returns:
            x: (B, T, n_embd)
        """
        B, T = token_types.shape
        dtype = task_embs.dtype

        is_bos = (token_types == 0).unsqueeze(-1)  # (B, T, 1)
        is_task = (token_types == 1).unsqueeze(-1)
        is_outcome = (token_types == 2).unsqueeze(-1)
        is_answer = (token_types == 3).unsqueeze(-1)

        # Project each type
        proj_task = self.task_proj(task_embs)  # (B, T, D)
        proj_outcome = self.outcome_proj(
            outcome_values.unsqueeze(-1).to(dtype)
        )  # (B, T, D)
        proj_answer = self.answer_proj(answer_embs)  # (B, T, D)

        # Select by type: BOS gets learned embedding, others get projections
        x = (
            is_bos.to(dtype) * self.bos_emb.to(dtype)
            + is_task.to(dtype) * proj_task
            + is_outcome.to(dtype) * proj_outcome
            + is_answer.to(dtype) * proj_answer
        )

        # Add type embeddings
        x = x + self.type_emb(token_types.long()).to(dtype)

        return x


class DFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)

        # Text encoder (set by load_encoder after checkpoint load)
        self.encoder = None

        # Embedding buffers (loaded at startup, not saved in checkpoints)
        self.register_buffer("task_embeddings", None, persistent=False)
        self.register_buffer("answer_embeddings", None, persistent=False)

        self.embedding = EmbeddingLayer(self.config)
        self.transformer = nn.ModuleDict(
            {
                "h": nn.ModuleList(
                    [
                        Block(self.config, layer_idx)
                        for layer_idx in range(self.config.n_layer)
                    ]
                ),
            }
        )
        # BCE prediction head — single logit per position
        self.prediction_head = nn.Linear(self.config.n_embd, 1, bias=False)

        # Rotary embeddings
        self.rotary_seq_len = self.config.block_size * 10
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(
        self, seq_len, head_dim, base=100000, device=None
    ):
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

    def load_encoder(self, model_name, max_length=512):
        """Create the TextEncoder module (weights loaded separately via load_state_dict)."""
        device = self.get_device()
        self.encoder = TextEncoder(model_name, max_length)
        self.encoder.to(device=device, dtype=torch.bfloat16)
        self.encoder.eval()

    def forward(
        self,
        token_types,
        task_indices,
        answer_indices,
        outcome_values,
        target_outcomes,
        kv_cache=None,
    ):
        """Full forward pass (used for checkpoint compatibility / training).

        Args:
            token_types: (B, T) int - 0=BOS, 1=TASK, 2=OUTCOME, 3=ANSWER
            task_indices: (B, T) int - task vocab index
            answer_indices: (B, T) int - answer vocab index
            outcome_values: (B, T) float - outcome at OUTCOME positions, -1 at BOS
            target_outcomes: (B, T) float - target at TASK positions, -1 elsewhere
        """
        B, T = token_types.shape

        # Look up pre-computed embeddings
        task_embs = self.task_embeddings[task_indices.long()]  # (B, T, n_input)
        answer_embs = self.answer_embeddings[answer_indices.long()]  # (B, T, n_input)

        # Pass through embedding layer
        x = self.embedding(token_types, task_embs, answer_embs, outcome_values)
        seq_len = x.size(1)

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0 : T0 + seq_len], self.sin[:, T0 : T0 + seq_len]

        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Loss mask: TASK positions (type 1) with valid targets
        valid_mask = (token_types == 1) & (target_outcomes >= 0)
        valid_hiddens = x[valid_mask]
        valid_targets = target_outcomes[valid_mask]

        logits = self.prediction_head(valid_hiddens).squeeze(-1).float()

        # BCE loss
        loss = F.binary_cross_entropy_with_logits(logits, valid_targets.float())

        return logits, loss, x.detach()


# ---------------------------------------------------------------------------
# Token-level inference helpers
# ---------------------------------------------------------------------------


def embed_tokens(
    model: DFM,
    token_types: torch.Tensor,
    task_embs: torch.Tensor,
    answer_embs: torch.Tensor,
    outcome_values: torch.Tensor,
) -> torch.Tensor:
    """Embed tokens via the model's EmbeddingLayer.

    Args:
        token_types: (B, T) int
        task_embs: (B, T, n_input) or (B, T, n_embd) if pre-projected
        answer_embs: (B, T, n_input) or (B, T, n_embd) if pre-projected
        outcome_values: (B, T) float
    Returns:
        (B, T, n_embd)
    """
    return model.embedding(token_types, task_embs, answer_embs, outcome_values)


def transformer_forward(model: DFM, x: torch.Tensor, kv_cache) -> torch.Tensor:
    """Run embedded token(s) through the transformer blocks and advance cache.

    Args:
        x: (B, T, n_embd) — already embedded tokens
        kv_cache: the learner's KV cache (KVCache or BatchedKVCache)
    Returns:
        hidden: (B, T, n_embd)
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


def predict_from_hiddens(model: DFM, hidden: torch.Tensor) -> torch.Tensor:
    """Apply BCE prediction head -> sigmoid probabilities.

    Args:
        hidden: (B, T, n_embd) or (N, n_embd)
    Returns:
        probabilities — same leading dims, last dim squeezed
    """
    logits = model.prediction_head(hidden).squeeze(-1).float()
    return torch.sigmoid(logits)
