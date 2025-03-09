from typing import Literal
import math

import einops
import torch
from torch import nn
from torch.nn import functional as F

from ...kernels import rope_apply, topk_attn
from .rmsnorm import RMSNorm
from ...misc.padding import pad_input, _upad_input

ATTN: Literal["sdpa", "flash", "sage"] = "flash"
TOP_K: int = 512
SPARSE_ATTENTION_START: float = 0.0
SPARSE_ATTENTION_RECOMPUTE: float = 0.0
SPARSE_ATTENTION_END: float = 0.0
POS_DICT = None
i = 0


@torch.no_grad()
def _dropout(
    attn: "Attention",
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    mask: torch.BoolTensor,
    apply_fn: callable,
    scale: float,
) -> tuple[bool, torch.Tensor]:
    global POS_DICT
    # from time import time

    # x0 = time()
    if attn.dropout:
        if attn.dropout_start or attn.dropout_recompute:
            POS_DICT = None

        attn_weights, POS_DICT = topk_attn(
            xq,
            xk,
            xv,
            mask,
            apply_fn,
            attn.head_dim,
            scale,
            TOP_K,
            POS_DICT,
            attn.n_heads,
            attn.n_kv_heads,
        )
        return False, attn_weights
    return True, None


def _do_attn(
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
    NEED_COMPUTE, attn_output = _dropout(
        attn, xq, xk, xv, attn_mask, apply_fn, softmax_scale
    )

    if not NEED_COMPUTE:
        return attn_output

    # print(xq.shape)

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


class JointAttention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        n_layers: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int = None,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layers = n_layers

        start, _start = 0, 1000.0
        end, _end = 0, 1000.0
        self.dropout = False
        if n_layers != 10000:
            for i in range(0, n_layers):
                perc = i / n_layers
                if not SPARSE_ATTENTION_START < perc < SPARSE_ATTENTION_END:
                    continue

                if i == layer_id:
                    self.dropout = True

                dist_start = abs(perc - SPARSE_ATTENTION_START)
                dist_recompute = abs(perc - SPARSE_ATTENTION_RECOMPUTE)

                if min(dist_start, _start) == dist_start:
                    start, _start = i, dist_start
                if min(dist_recompute, _end) == dist_recompute:
                    end, _end = i, dist_recompute

            self.dropout_start = start == layer_id
            self.dropout_recompute = end == layer_id
            if self.dropout:
                print(f"Layer {layer_id} {self.dropout_start} {self.dropout_recompute}")

        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        # from time import time

        # t0 = time()
        rbsz, _ = x_mask.shape
        if x.dim() == 2:
            x = x.view(rbsz, x.shape[0] // rbsz, -1).contiguous()
        bsz, seqlen, _ = x.shape
        # print(f"JointAttention {x.shape}")

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = rope_apply(xq, xk, freqs_cis)

        softmax_scale = math.sqrt(1 / self.head_dim)
        if ATTN == "sdpa":
            n_rep = self.n_heads // self.n_local_kv_heads
            if n_rep >= 1:
                xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        output = _do_attn(
            self,
            xq,
            xk,
            xv,
            x_mask,
            lambda x: x.bool()
            .view(bsz, 1, 1, seqlen)
            .expand(-1, self.n_heads, seqlen, -1),
            bsz,
            seqlen,
            softmax_scale,
        )
        output = output.flatten(-2)
        ret = self.wo(output)
        # print(f"JointAttention {time() - t0:.4f}")
        return ret


class Attention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        n_layers: int,
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
        self.layer_id = layer_id
        self.n_layers = n_layers

        start, _start = 0, 1000.0
        end, _end = 0, 1000.0
        self.dropout = False
        if n_layers != 10000:
            for i in range(0, n_layers):
                perc = i / n_layers
                if not SPARSE_ATTENTION_START < perc < SPARSE_ATTENTION_END:
                    continue

                if i == layer_id:
                    self.dropout = True

                dist_start = abs(perc - SPARSE_ATTENTION_START)
                dist_recompute = abs(perc - SPARSE_ATTENTION_RECOMPUTE)

                if min(dist_start, _start) == dist_start:
                    start, _start = i, dist_start
                if min(dist_recompute, _end) == dist_recompute:
                    end, _end = i, dist_recompute

            self.dropout_start = start == layer_id
            self.dropout_recompute = end == layer_id

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
        xq, xk = rope_apply(xq, xk, freqs_cis)

        output = _do_attn(
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
            output_y = _do_attn(
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
