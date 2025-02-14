from typing import Literal

import einops
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._triton import has_triton

from ...misc.padding import pad_input, _upad_input


# if has_triton():
#     from ...kernels.rope import apply_merged_rotary_embedding
# else:


def apply_merged_rotary_embedding(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_complex = torch.view_as_complex(xq_)
    xk_complex = torch.view_as_complex(xk_)

    freqs_cis = freqs_cis.unsqueeze(2)

    xq_out = xq_complex * freqs_cis
    xk_out = xk_complex * freqs_cis

    xq_out = torch.view_as_real(xq_out).flatten(-2)
    xk_out = torch.view_as_real(xk_out).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


ATTN: Literal["sdpa", "flash", "sage", "flash-int8"] = "flash-int8"


# TODO: bring out to helper class
def do_attn(
    attn: "Attention",
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    attn_mask: torch.BoolTensor,
    apply_fn: callable,
    batch: int,
    seq_len: int,
    softmax_scale: float = None,
) -> torch.Tensor:
    if ATTN != "sdpa" and xq.dtype in [torch.bfloat16, torch.float16]:
        (
            query_states,
            key_states,
            value_states,
            indices_q,
            cu_seq_lens,
            max_seq_lens,
        ) = _upad_input(attn, xq, xk, xv, attn_mask, seq_len)

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        if ATTN == "flash":
            from flash_attn import flash_attn_varlen_func

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=0.0,
                causal=False,
                softmax_scale=softmax_scale,
                # softcap=30,
            )
        elif ATTN == "flash-int8":
            from ...kernels.fa import _Attention

            attn_output_unpad = _Attention.apply(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_in_batch_q,
                max_seqlen_in_batch_k,
                softmax_scale,
            )
        else:
            from sageattention import sageattn_varlen

            attn_output_unpad = sageattn_varlen(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                sm_scale=softmax_scale,
            )
        return pad_input(attn_output_unpad, indices_q, batch, seq_len)

    return F.scaled_dot_product_attention(
        xq.permute(0, 2, 1, 3),
        xk.permute(0, 2, 1, 3),
        xv.permute(0, 2, 1, 3),
        attn_mask=apply_fn(attn_mask),
        scale=softmax_scale,
    ).permute(0, 2, 1, 3)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int = None,
        qk_norm: bool = False,
        y_dim: int = 0,
        base_seqlen: int = None,
        proportional_attn: bool = False,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 384,
        qk_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.qk_norm = qk_norm
        self.y_dim = y_dim
        self.base_seqlen = base_seqlen
        self.proportional_attn = proportional_attn
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings

        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)

        if y_dim > 0:
            self.wk_y = nn.Linear(y_dim, self.n_kv_heads * self.head_dim, bias=False)
            self.wv_y = nn.Linear(y_dim, self.n_kv_heads * self.head_dim, bias=False)
            self.gate = nn.Parameter(torch.zeros(n_heads))

        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        if qk_norm:
            if not qk_bias:
                self.q_norm = nn.LayerNorm(self.head_dim, bias=qk_bias)
                self.k_norm = nn.LayerNorm(self.head_dim, bias=qk_bias)
            else:
                self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim, bias=qk_bias)
                self.k_norm = nn.LayerNorm(
                    self.n_kv_heads * self.head_dim, bias=qk_bias
                )
            if y_dim > 0:
                self.ky_norm = nn.LayerNorm(
                    self.n_kv_heads * self.head_dim, eps=1e-6, bias=qk_bias
                )
            else:
                self.ky_norm = nn.Identity()
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.ky_norm = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.BoolTensor,
        freqs_cis: torch.Tensor,
        y: torch.Tensor = None,
        y_mask: torch.BoolTensor = None,
        init_cache: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        xq: torch.Tensor = self.wq(x)
        xk: torch.Tensor = self.wk(x)
        xv: torch.Tensor = self.wv(x)

        if x_mask is None:
            x_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=x.device)
        input_dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(batch, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch, seq_len, self.n_kv_heads, self.head_dim)

        if self.n_kv_heads != self.n_heads:
            # kv-grouping
            n_repeat = self.n_heads // self.n_kv_heads
            xk = xk.repeat_interleave(n_repeat, dim=2)
            xv = xv.repeat_interleave(n_repeat, dim=2)

        freqs_cis = freqs_cis.to(xq.device)
        xq, xk = apply_merged_rotary_embedding(xq, xk, freqs_cis)

        output = do_attn(
            self,
            xq,
            xk,
            xv,
            x_mask,
            lambda x: x.bool()
            .view(batch, 1, 1, seq_len)
            .expand(-1, self.n_heads, seq_len, -1),
            batch,
            seq_len,
        ).to(input_dtype)

        if hasattr(self, "wk_y"):
            yk = self.ky_norm(self.wk_y(y)).view(
                batch, -1, self.n_kv_heads, self.head_dim
            )
            yv = self.wv_y(y).view(batch, -1, self.n_kv_heads, self.head_dim)
            n_repeat = self.n_heads // self.n_kv_heads
            if n_repeat >= 1:
                yk = einops.repeat(yk, "b l h d -> b l (repeat h) d", repeat=n_repeat)
                yv = einops.repeat(yv, "b l h d -> b l (repeat h) d", repeat=n_repeat)
            output_y = do_attn(
                self,
                xq,
                yk,
                yv,
                y_mask,
                lambda x: x.bool()
                .view(batch, 1, 1, -1)
                .expand(batch, self.n_heads, seq_len, -1),
                batch,
                seq_len,
            )
            output_y = output_y * self.gate.tanh().view(1, 1, -1, 1)
            output = output + output_y

        output = output.flatten(-2)
        output = self.wo(output)

        return output.to(input_dtype)
