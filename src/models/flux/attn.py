import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..common.attn import ATTN
from ..common.nextdit import TimestepEmbedder
from ..common.rmsnorm import RMSNorm
from ...kernels import rope_apply


class FluxTimestepEmbedder(TimestepEmbedder):
    def __init__(self, hidden_size: int, frequency_embeddings_size: int = 256):
        super().__init__(hidden_size, frequency_embeddings_size, True)

    def timestep_embedding(
        self, t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        t = 1000.0 * t
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding


def _do_attn(xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor) -> torch.Tensor:
    if ATTN == "flash":
        from flash_attn import flash_attn_func

        x = flash_attn_func(xq, xk, xv).transpose(1, 2)
    elif ATTN == "sage":
        from sageattention import sageattn

        x = sageattn(xq, xk, xv).transpose(1, 2)
    else:
        x = F.scaled_dot_product_attention(xq, xk, xv).transpose(1, 2)

    return x.reshape(*x.shape[:-2], -1)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.out = nn.Linear(dim, dim)
        self.K = 3
        self.H = self.num_heads
        self.KH = self.K * self.H

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)

        B, L, D = x.shape
        q, k, v = qkv.reshape(B, L, self.K, self.H, D // self.KH).permute(2, 0, 3, 1, 4)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = rope_apply(q, k, freqs_cis)
        x = _do_attn(q, k, v)

        x = self.out(x)
        return x
