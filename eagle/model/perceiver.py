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

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


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
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout_p
        )

        out = out.transpose(1, 2).contiguous().view(B, L_q, D)
        return self.wo(out)


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
        dim: perceiver latent width (typically the draft model's hidden size).
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
        self.n_latents = n_latents

        self.layer_encoding = nn.Parameter(torch.randn(1, num_target_layers, 1, dim))
        self.latent_queries = nn.Parameter(torch.randn(1, n_latents, dim))
        self.expert2latent = nn.Linear(d_target, dim, bias=False)

        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, S, d_target] (target layers L, sequence length S)

        Returns:
            [B, S * n_latents, dim], where latents for token j occupy columns
            [j * n_latents : (j + 1) * n_latents].
        """
        batch_size, num_layers, seq_len, _ = hidden_states.shape
        _, n_latents, dim = self.latent_queries.shape

        # Send to perceiver dim and add layer encoding
        hidden_states = self.expert2latent(hidden_states)
        hidden_states = hidden_states + self.layer_encoding

        # [B, L, S, D] --> [B*S, L, D]
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * seq_len, num_layers, dim)

        latent_queries = self.latent_queries.expand(batch_size * seq_len, -1, -1)
        for layer in self.layers:
            kv = torch.cat([hidden_states, latent_queries], dim=1)
            latent_queries = layer(q=latent_queries, kv=kv)

        # [B*S, n_latents, D] --> [B, S*n_latents, D]
        return latent_queries.view(batch_size, seq_len * n_latents, dim)
