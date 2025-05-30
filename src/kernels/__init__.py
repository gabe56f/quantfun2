import math

import torch
import numpy as np

from .module import implemented_as, get_source_module


def _apply_rope_triton(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    from .triton.rope import rope_triton

    return rope_triton(xq, xk, freqs_cis)


def _apply_inplace_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    from .triton.softmax import inplace_tiled_softmax

    assert dim == -1, "Only dim=-1 is supported"

    return inplace_tiled_softmax(x)


def _apply_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    from .triton.rmsnorm import apply_rmsnorm

    return apply_rmsnorm(x, weight, eps)


def _apply_partial_matmul_triton(x: torch.Tensor, w: torch.Tensor, index: torch.Tensor):
    from .triton.ras import _partially_linear

    return _partially_linear(x, w, index)


def _apply_partial_rope_cuda(xq, xk, freqs_cis):
    import triton

    apply_kernel = get_source_module("rope")

    B, N, H, D = xq.shape

    xq_out = torch.empty_like(xq)
    xk_out = torch.empty_like(xk)

    thread_number = 128
    block_n = thread_number // H

    # fmt: off
    apply_kernel(
        xq.view(torch.int16),
        xk.view(torch.int16),
        torch.view_as_real(freqs_cis),
        xq_out.view(torch.int16),
        xk_out.view(torch.int16),
        np.int32(N), np.int32(H), np.int32(D),
        grid=(triton.cdiv(N, block_n), B, 1),
        block=(H, block_n, 1),
    )
    # fmt: on

    return rope_apply(xq, xk, freqs_cis)


@implemented_as(triton_func=_apply_partial_matmul_triton)
def matmul_partial(x: torch.Tensor, w: torch.Tensor, index: torch.Tensor):
    return torch.matmul(x, w)[:, index]


@implemented_as(cuda_func=_apply_partial_rope_cuda)
def rope_apply_partial(): ...


@implemented_as(triton_func=_apply_rmsnorm)
def rmsnorm_apply(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    def _norm(x: torch.Tensor) -> torch.Tensor:
        nonlocal eps
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    output = _norm(x.float()).type_as(x)
    return output * weight


@implemented_as(triton_func=_apply_inplace_softmax)
def inplace_softmax(x: torch.Tensor, dim: int = -1):
    # assert dim == -1, "Only dim=-1 is supported"

    return torch.softmax(x, dim=dim, out=x)


# @implemented_as(triton_func=_apply_ffn_forward_triton)
def ffn_forward(
    x: torch.Tensor,
    c: torch.Tensor,
    adaLN_weight: torch.Tensor,
    adaLN_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    bias: bool,
    hidden_size: int,
    num_patches: int,
    out_channels: int,
) -> torch.Tensor:
    torch.nn.functional.silu(c, inplace=True)
    scale = torch.nn.functional.linear(c, adaLN_weight, adaLN_bias)

    def modulate(x: torch.Tensor, scale: float) -> torch.Tensor:
        return x * (1 + scale)

    # print(hidden_size)
    # print(x.shape, scale.shape)
    x = torch.nn.functional.layer_norm(x, (hidden_size,), eps=1e-6)
    # print(x.shape, scale.shape)
    x = modulate(x, scale.unsqueeze(1) if bias else scale)
    x = torch.nn.functional.linear(x, linear_weight, linear_bias)
    # print(x.shape)
    # print("---")
    return x


# TODO: implement this in triton or cuda
def topk(x: torch.Tensor, k: int):
    return torch.topk(x, k, dim=-1).indices


# TODO: implement this in cuda maybe
@implemented_as(triton_func=_apply_rope_triton)
def rope_apply(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
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


# @implemented_as(triton_func=_apply_topk_fa_attn_triton)
def topk_attn(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    mask: torch.BoolTensor,
    fn: callable,
    HEAD_DIM: int,
    scale: float = None,
    TOP_K: int = 256,
    POS_DICT: torch.BoolTensor = None,
    N_HEADS: int = 32,
    N_KV_HEADS: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_rep = N_HEADS // N_KV_HEADS
    if n_rep >= 1:
        xk = xk.unsqueeze_(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        xv = xv.unsqueeze_(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

    # xq, xk, xv: [batch_size, n_queries, n_heads, head_dim]
    # -> x_: [batch_size, n_heads, n_queries, head_dim]
    xq.transpose_(1, 2)
    xk.transpose_(1, 2)
    xv.transpose_(1, 2)

    if scale is None:
        scale = 1.0 / math.sqrt(HEAD_DIM)

    torch.matmul(xq, xk.transpose_(2, 3), out=xq)
    xq *= scale

    if mask is not None:
        xq += fn(mask)[:, :, :, : xq.shape[-2]]

    if POS_DICT is None:
        last_dim_size = xq.size(-1)
        token_budget = min(last_dim_size, TOP_K)
        _, top_k_indices = torch.topk(xq, token_budget, sorted=False)
        POS_DICT = torch.zeros_like(xq, dtype=torch.bool).scatter_(
            -1, top_k_indices, True
        )
    else:
        min_val = torch.finfo(xq.dtype).min
        xq.masked_fill_(~POS_DICT.to(xq.device), min_val)

    xq = inplace_softmax(xq, -1)

    torch.matmul(xq, xv, out=xq)
    xq.transpose_(1, 2)

    return xq, POS_DICT
