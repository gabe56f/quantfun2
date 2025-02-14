"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton
import triton.language as tl


configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [32, 64, 128, 256]
    for BN in [32, 64, 128, 256]
    for s in ([3, 4, 7])
    for w in [4, 8]
]


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    q_scale,
    kv_len,
    K_ptr,
    K_scale_ptr,
    V_ptr,
    stride_kn,
    stride_vn,
    start_m,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N: tl.constexpr,
    H_K: tl.constexpr,
):
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[:, None] < (kv_len - start_n)  # Transposed mask for k

        # --- [BLOCK_N, HEAD_DIM] ---
        k_ptr = tl.make_block_ptr(
            base=K_ptr,
            shape=(N, HEAD_DIM),
            strides=(stride_kn, 1),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),  # (0, 1)
        )
        k = tl.load(k_ptr, mask=k_mask, other=0.0)  # Load with mask and default value

        k_scale_ptr = tl.make_block_ptr(
            base=K_scale_ptr,
            shape=(H_K,),
            strides=(1,),
            offsets=(0,),
            block_shape=(H_K,),
            order=(0,),
        )

        k_scale = tl.load(k_scale_ptr)  # Load scale
        qk = tl.dot(q, tl.trans(k))  # .to(tl.float32)  # Dot product and scale
        qk = tl.where(
            offs_n[None, :] < kv_len - start_n, qk, float("-inf")
        )  # apply mask for heads that are too short
        qk = qk * q_scale * k_scale

        m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Update max
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)  # Exponential
        l_ij = tl.sum(p, 1)  # Sum of p

        alpha = tl.math.exp2(m_i - m_ij)  # Scaling factor
        l_i = l_i * alpha + l_ij  # Update l_i

        acc = acc * alpha[:, None]  # Scale accumulator

        # -- [BLOCK_M, BLOCK_N] --
        v_ptr = tl.make_block_ptr(
            base=V_ptr,
            shape=(N, HEAD_DIM),
            strides=(stride_vn, 1),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),  # (0, 1)
        )
        v = tl.load(v_ptr, mask=k_mask, other=0.0)
        p = p.to(tl.float16)  # p.to(v.dtype)
        acc += tl.dot(p, v)  # Update accumulator (no type conversion needed)
        m_i = m_ij  # Update m_i

        K_scale_ptr += H_K

    return acc, l_i


@triton.autotune(configs, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    cu_seqlens_q,
    cu_seqlens_k,
    Q_scale,
    K_scale,
    cu_seqlens_q_scale,
    cu_seqlens_k_scale,
    Out,
    stride_qh,
    stride_qn,
    stride_kh,
    stride_kn,
    stride_vh,
    stride_vn,
    stride_oh,
    stride_on,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
    cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
    qo_len = cu_seqlens_q_end - cu_seqlens_q_start
    if (start_m * BLOCK_M) >= qo_len:
        return

    cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
    cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
    kv_len = cu_seqlens_k_end - cu_seqlens_k_start

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # --- Load Q ---
    q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(qo_len, HEAD_DIM),  # Shape of the Q tensor
        strides=(stride_qn, 1),  # Strides of the Q tensor
        offsets=(
            cu_seqlens_q_start + start_m * BLOCK_M,
            off_h * HEAD_DIM,
        ),  # Initial offsets
        block_shape=(BLOCK_M, HEAD_DIM),  # Shape of the block to load
        order=(1, 0),  # Row-major
    )

    q_mask = offs_m[:, None] < qo_len
    q = tl.load(q_block_ptr, mask=q_mask, other=0.0)  # Load with mask

    cu_seq_lens_q_scale_start = tl.load(cu_seqlens_q_scale + off_z)
    q_scale_offset = cu_seq_lens_q_scale_start + off_h

    q_scale_ptr = tl.make_block_ptr(
        base=Q_scale,
        shape=(qo_len,),
        strides=(1,),
        offsets=(q_scale_offset,),
        block_shape=(1,),
        order=(0,),
    )

    q_scale = tl.load(q_scale_ptr)  # Load Q scale

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Same for K, K_scale, V -- Use block pointers with proper offsets and masks
    acc, l_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        q_scale,
        kv_len,
        K,  # Pass base pointers
        K_scale,  # Pass base pointer
        V,  # Pass base pointers
        stride_kn,
        stride_vn,
        start_m,
        H,  # // num_kv_groups,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        offs_m,
        offs_n,
        kv_len,  # Shape of K and V
        H // num_kv_groups,
    )

    acc = acc / l_i[:, None]  # .to(acc.dtype)

    # --- Store Output ---
    o_block_ptr = tl.make_block_ptr(
        base=Out,
        shape=(qo_len, HEAD_DIM),
        strides=(stride_on, 1),
        offsets=(cu_seqlens_q_start + start_m * BLOCK_M, off_h * HEAD_DIM),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(o_block_ptr, acc, mask=q_mask)  # Store with mask


def forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    q_scale,
    k_scale,
    cu_seqlens_q_scale,
    cu_seqlens_k_scale,
    output_dtype=torch.float16,
):
    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    b = cu_seqlens_q.shape[0] - 1
    s, h_qo, head_dim = q.shape
    _, h_kv, _ = k.shape

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    def grid(META):
        return (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), h_qo, b)

    _attn_fwd[grid](
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        q_scale,
        k_scale,
        cu_seqlens_q_scale,
        cu_seqlens_k_scale,
        o,
        q.stride(1),
        q.stride(0),
        k.stride(1),
        k.stride(0),
        v.stride(1),
        v.stride(0),
        o.stride(1),
        o.stride(0),
        h_qo,
        s,
        num_kv_groups,
        HEAD_DIM=HEAD_DIM_K,
    )
    return o
