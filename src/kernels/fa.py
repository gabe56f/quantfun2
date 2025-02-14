import triton
import triton.language as tl
import torch


def get_configs():
    configs = []
    for warp in [4, 8]:
        for m in [256, 128, 64, 32, 16]:
            for n in [64, 32, 16]:
                for waves in range(1, 5):
                    for v in [True, False]:
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": m,
                                    "BLOCK_N": n,
                                    "PRE_LOAD_V": v,
                                    "waves_per_eu": waves,
                                },
                                num_stages=1,
                                num_warps=warp,
                            )
                        )
    return configs


@triton.jit
def cdiv(a, b):
    return (a + b - 1) // b


@triton.jit
def load(ptr, first, second, padding):
    if first and second:
        tensor = tl.load(ptr, boundary_check=(0, 1), padding_option=padding)
    elif first:
        tensor = tl.load(ptr, boundary_check=(0,), padding_option=padding)
    elif second:
        tensor = tl.load(ptr, boundary_check=(1,), padding_option=padding)
    else:
        tensor = tl.load(ptr)
    return tensor


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    q_scale,
    K_block_ptr,
    K_block_scale_ptr,
    V_block_ptr,
    v_scale,
    actual_seqlen_k,
    block_min,
    block_max,
    n_extra_tokens,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
):
    for start_n in range(block_min, block_max, BLOCK_N):
        k = load(K_block_ptr, PADDED_HEAD, MASK_STEPS and (n_extra_tokens != 0), "zero")
        k_scale = tl.load(K_block_scale_ptr)
        if PRE_LOAD_V:
            v = load(
                V_block_ptr, MASK_STEPS and (n_extra_tokens != 0), PADDED_HEAD, "zero"
            )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if MASK_STEPS:
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))

        qk = tl.dot(q, k).to(tl.float32)
        qk = qk * q_scale[:, None]
        qk = qk * k_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        p = p.to(tl.float16)
        p = p * 127
        p = (p + 0.5).to(tl.int8)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha * l_ij
        alpha = acc * alpha[:, None]

        if not PRE_LOAD_V:
            v = load(
                V_block_ptr, MASK_STEPS and (n_extra_tokens != 0), PADDED_HEAD, "zero"
            )
        tmp = tl.dot(p, v)
        tmp = tmp.to(tl.float32)
        tmp = tmp * v_scale / 127
        acc += tmp

        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        K_block_scale_ptr = tl.advance(K_block_scale_ptr, (BLOCK_N,))
    return (acc, l_i, m_i)


# @triton.autotune(configs=get_configs(), key=["BLOCK_DMODEL"])
@triton.jit
def attn_fwd(
    Q,
    Q_scale,
    K,
    K_scale,
    V,
    V_scale,
    sm_scale,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_sq1,
    stride_sq2,
    stride_sq3,
    stride_sk1,
    stride_sk2,
    stride_sk3,
    stride_v1,
    stride_v2,
    cu_seqlens_q,
    cu_seqlens_k,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_n = tl.arange(0, BLOCK_N)

    cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
    cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
    seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
    if start_m * BLOCK_M > seqlen_q:
        # early return if too small seqlen
        return
    cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
    cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
    seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start

    n_blocks = cdiv(seqlen_q, BLOCK_M)
    GROUP_SIZE: tl.constexpr = HQ // HK
    off_h_k = off_h_q // GROUP_SIZE if GROUP_SIZE != 1 else off_h_q

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    padded_head = ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL

    q_offset = off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    k_offset = off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(ACTUAL_BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    v_offset = off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seqlen_q, seqlen_k),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    qs_offset = (
        off_z * stride_sq1 + off_h_q * stride_sq2 + cu_seqlens_q_start * stride_sq3
    )
    Q_block_scale_ptr = tl.make_block_ptr(
        base=Q_scale + qs_offset,
        shape=(seqlen_q,),
        strides=(stride_sq3,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    ks_offset = (
        off_z * stride_sk1 + off_h_k * stride_sk2 + cu_seqlens_k_start * stride_sk3
    )
    K_block_scale_ptr = tl.make_block_ptr(
        base=K_scale + ks_offset,
        shape=(seqlen_k,),
        strides=(stride_sk3,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )
    vs_offset = off_z * stride_v1 + off_h_k * stride_v2

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale: tl.constexpr = sm_scale * 1.44269504089

    q = load(Q_block_ptr, True, padded_head, "zero")
    q_scale = tl.load(Q_block_scale_ptr)
    v_scale = tl.load(V_scale + vs_offset)

    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)

    padded_block_k = n_extra_tokens != 0
    masked_blocks = min(padded_block_k, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        resp = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_scale,
            K_block_ptr,
            K_block_scale_ptr,
            V_block_ptr,
            v_scale,
            seqlen_k,
            block_min,
            block_max,
            0,
            BLOCK_M,
            BLOCK_N,
            offs_n,
            PRE_LOAD_V,
            False,
            padded_head,
        )
        acc = resp[0]
        l_i = resp[1]
        m_i = resp[2]
        block_min = block_max
        block_max = n_blocks * BLOCK_N
    tl.debug_barrier()

    if masked_blocks > 0:
        K_block_ptr = tl.advance(K_block_ptr, (0, n_full_blocks * BLOCK_N))
        K_block_scale_ptr = tl.advance(K_block_scale_ptr, (n_full_blocks * BLOCK_N,))
        V_block_ptr = tl.advance(V_block_ptr, (n_full_blocks * BLOCK_N, 0))

        resp = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_scale,
            K_block_ptr,
            K_block_scale_ptr,
            V_block_ptr,
            v_scale,
            seqlen_k,
            block_min,
            block_max,
            n_extra_tokens,
            BLOCK_M,
            BLOCK_N,
            offs_n,
            PRE_LOAD_V,
            True,
            padded_head,
        )
        acc = resp[0]
        l_i = resp[1]
        m_i = resp[2]
    acc = acc = l_i[:, None]
    acc = acc.to(Out.type.element_ty)
    o_offset = off_z * stride_oz + cu_seqlens_q_start * stride_om + off_h_q * stride_oh
    Out_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(Out_block_ptr, acc, boundary_check=(0, 1))


def quant_pertoken(X: torch.Tensor) -> tuple:
    X_max, _ = torch.abs(X).max(dim=-1)
    X_scale = X_max / 127
    ret = torch.round(X / X_scale[:, :, None]).to(torch.int8)
    return ret, X_scale


def quant_pertensor(X: torch.Tensor) -> tuple:
    X_max, _ = torch.abs(X).max(dim=-1)
    X_max, _ = torch.max(X_max, dim=-1)
    X_scale = X_max / 127
    ret = torch.round(X / X_scale[:, None, None]).to(torch.int8)
    return ret, X_scale


class _Attention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlens_q,
        max_seqlens_k,
        sm_scale=1.0,
    ):
        o = torch.empty_like(q, dtype=v.dtype)
        # q (8192, 24, 96)
        # k (8192, 8, 96)
        # v (8192, 8, 96)
        # cu_seqlens_q ((3))
        # cu_seqlens_k ((3))
        q, q_scale = quant_pertoken(q)
        k, k_scale = quant_pertoken(k)
        v, v_scale = quant_pertensor(v)

        total_q, nheads_q, head_size = q.shape
        total_k, nheads_k, _ = k.shape
        batch = len(cu_seqlens_q) - 1
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
        qs_strides = (0, q_scale.stride(0), q_scale.stride(1))
        ks_strides = (0, k_scale.stride(0), k_scale.stride(1))
        vs_strides = (0, v_scale.stride(0))

        unpadded_head_dims = [32, 64, 128, 256]
        if head_size not in unpadded_head_dims:
            padded_d_model = None
            for i in unpadded_head_dims:
                if i > head_size:
                    padded_d_model = i
                    break
        else:
            padded_d_model = head_size

        def grid(META):
            return (triton.cdiv(max_seqlens_q, META["BLOCK_M"]), nheads_q, batch)

        attn_fwd[grid](
            q,
            q_scale,
            k,
            k_scale,
            v,
            v_scale,
            sm_scale,
            o,
            *q_strides,
            *k_strides,
            *v_strides,
            *o_strides,
            *qs_strides,
            *ks_strides,
            *vs_strides,
            cu_seqlens_q,
            cu_seqlens_k,
            HQ=nheads_q,
            HK=nheads_k,
            ACTUAL_BLOCK_DMODEL=head_size,
            BLOCK_DMODEL=padded_d_model,
            BLOCK_M=256,
            BLOCK_N=64,
            PRE_LOAD_V=True,
        )

        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = head_size

        return o
