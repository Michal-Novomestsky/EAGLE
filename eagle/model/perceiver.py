"""
Perceiver resampler for EAGLE.

Adapted from the PerceiverResampler in
https://github.com/Michal-Novomestsky/Pseudodistillation/tree/michal/decoder-retrain
(src/model/adapter.py, src/model/attention.py).

Instead of concatenating a few target-model hidden states and compressing them
with a single FC block (EAGLE-3), the perceiver resampler consumes the *entire*
residual stream of the target model: for each token position it treats the stack
of all layer hidden states as a short sequence and cross-attends learnable
latent queries over it, emitting ``n_latents`` tokens per position (default 1,
i.e. the whole residual stream is flattened into a single token).
"""

import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

# ---------------------------------------------------------------------------
# Forward-stat logging, for diagnosing training dynamics vs. the EAGLE-3 FC.
# Set PERCEIVER_LOG_EVERY=N to emit stats every N training forwards
# (0 disables). Lines are prefixed "PERCEIVER_STATS " and printed to stdout
# (captured by SLURM into the .out file), and optionally appended as JSONL to
# $PERCEIVER_LOG_FILE if set.
# ---------------------------------------------------------------------------
_LOG_EVERY = int(os.environ.get("PERCEIVER_LOG_EVERY", "50"))
_LOG_FILE = os.environ.get("PERCEIVER_LOG_FILE")
_fwd_count = 0


def _rms(x: torch.Tensor) -> float:
    return x.float().pow(2).mean().sqrt().item()


def _emit(stats: dict) -> None:
    line = "PERCEIVER_STATS " + json.dumps(stats)
    print(line, flush=True)
    if _LOG_FILE:
        with open(_LOG_FILE, "a") as f:
            f.write(json.dumps(stats) + "\n")


class RMSNorm(nn.Module):
    """Local RMSNorm so we do not depend on a recent torch for nn.RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class SwiGLU(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        # Set by PerceiverResampler on logging steps; stats land in _stats.
        self._collect_stats = False
        self._stats: dict = {}

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L_q, D = x_q.shape
        _, L_kv, _ = x_k.shape

        q = self.wq(x_q)
        k = self.wk(x_k)
        v = self.wv(x_v)

        q = q.view(B, L_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.n_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0

        if self._collect_stats:
            # Recompute logits under no_grad purely for diagnostics; this runs
            # only every PERCEIVER_LOG_EVERY forwards, so the cost is negligible.
            with torch.no_grad():
                QK = q.float() @ k.float().transpose(-2, -1) / math.sqrt(self.head_dim)
                probs = F.softmax(QK, dim=-1)
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1).mean()
                # max entropy over L_kv keys = log(L_kv); near 1 => uniform
                self._stats = {
                    "qk_std": QK.std().item(),
                    "qk_absmax": QK.abs().max().item(),
                    "softmax_entropy_frac": (entropy / math.log(L_kv)).item(),
                }

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout_p
        )

        out = out.transpose(1, 2).contiguous().view(B, L_q, D)
        out = self.wo(out)
        if self._collect_stats:
            with torch.no_grad():
                self._stats["attn_out_rms"] = _rms(out)
        return out


class TransformerBlock(nn.Module):
    """Single pre-norm transformer block (cross-attention from q to kv, then FFN)."""

    def __init__(self, dim: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.attention = MultiHeadAttention(dim, n_heads, dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, d_ff, dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = q + self.attention(
            self.attention_norm(q),
            self.attention_norm(kv),
            self.attention_norm(kv),
            attention_mask,
        )
        return x + self.ffn(self.ffn_norm(x))


class PerceiverResampler(nn.Module):
    """
    Flattens the target model's residual stream into ``n_latents`` tokens per
    position.

    Args:
        num_target_layers: number of target-model layers whose hidden states are
            stacked as the perceiver input (L).
        d_target: hidden size of the target model.
        dim: internal perceiver latent width.
        draft_hidden_size: output width passed to the draft decoder (via
            ``latent2decoder``).
        n_heads: attention heads per block.
        d_ff: SwiGLU FFN hidden size.
        num_layers: number of TransformerBlocks (default 6, as in Flamingo).
        n_latents: latents per token position (default 1 = a single token).
        dropout: dropout inside attention/FFN (inactive at eval).
    """

    def __init__(
        self,
        num_target_layers: int,
        d_target: int,
        dim: int,
        draft_hidden_size: int,
        n_heads: int,
        d_ff: int,
        num_layers: int = 6,
        n_latents: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_target_layers = num_target_layers
        self.d_target = d_target
        self.dim = dim
        self.draft_hidden_size = draft_hidden_size
        self.n_latents = n_latents

        self.pre_norm = RMSNorm(d_target)
        self.layer_encoding = nn.Parameter(torch.randn(1, num_target_layers, 1, d_target))
        self.latent_queries = nn.Parameter(torch.randn(1, n_latents, dim))
        self.expert2latent = nn.Linear(d_target, dim, bias=False)
        self.latent2decoder = nn.Linear(dim, draft_hidden_size, bias=False)

        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, S, d_target] (target layers L, sequence length S)

        Returns:
            [B, S * n_latents, draft_hidden_size], where latents for token j
            occupy columns [j * n_latents : (j + 1) * n_latents].
        """
        batch_size, num_layers, seq_len, _ = hidden_states.shape
        _, n_latents, dim = self.latent_queries.shape

        global _fwd_count
        log_this = self.training and _LOG_EVERY > 0 and _fwd_count % _LOG_EVERY == 0
        _fwd_count += 1

        if log_this:
            with torch.no_grad():
                input_rms = _rms(hidden_states)

        # Project to perceiver dim, normalize, and add layer encoding
        hidden_states = self.pre_norm(hidden_states)
        hidden_states = hidden_states + self.layer_encoding
        hidden_states = self.expert2latent(hidden_states)

        # [B, L, S, D] --> [B*S, L, D]
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * seq_len, num_layers, dim)

        latent_queries = self.latent_queries.expand(batch_size * seq_len, -1, -1)
        if log_this:
            with torch.no_grad():
                anchor = latent_queries.detach().clone()
                kv_rms = _rms(hidden_states)

        layer_stats = []
        for layer in self.layers:
            kv = torch.cat([hidden_states, latent_queries], dim=1)
            layer.attention._collect_stats = log_this
            latent_queries = layer(q=latent_queries, kv=kv)
            if log_this:
                layer_stats.append(layer.attention._stats)
                layer.attention._collect_stats = False

        if log_this:
            with torch.no_grad():
                signal = latent_queries - anchor
                stats = {
                    "fwd": _fwd_count - 1,
                    "input_rms": input_rms,
                    "kv_rms_prenorm": kv_rms,
                    "layer_enc_rms": _rms(self.layer_encoding),
                    "anchor_rms": _rms(anchor),
                    "out_rms": _rms(latent_queries),
                    # Position-dependent signal relative to the constant latent
                    # anchor: ~0 means the draft sees mostly a constant vector.
                    "signal_over_anchor": _rms(signal) / max(_rms(anchor), 1e-8),
                    "layers": layer_stats,
                }
            _emit(stats)

        # [B*S, n_latents, dim] --> [B, S*n_latents, draft_hidden_size]
        latent_queries = self.latent2decoder(latent_queries)
        return latent_queries.view(batch_size, seq_len * n_latents, self.draft_hidden_size)
