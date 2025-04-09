import torch
import triton
import triton.language as tl


def _configs():
    configs = []
    for BLOCK_SIZE in [64, 128]:
        for num_warps in [1, 2]:
            configs.append(
                triton.Config(
                    {"BLOCK_SIZE": BLOCK_SIZE},
                    num_warps=num_warps,
                    num_stages=1,
                )
            )
    return configs


@triton.autotune(configs=_configs(), key=["seq_len", "num_heads"], rep=30, warmup=10)
@triton.jit
def rope_kernel_inplace(
    x_ptr,
    cos_ptr,
    sin_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    stride_batch,
    stride_seq,
    stride_head,
    stride_batch_theta,
    stride_seq_theta,
    num_pairs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # Total number of pairs to process per head
    total_pairs: tl.constexpr = batch_size * seq_len * num_heads * num_pairs

    # Compute base offset for this block
    base_pair_idx = pid * BLOCK_SIZE
    pair_offsets = base_pair_idx + tl.arange(0, BLOCK_SIZE)

    # Early exit if all threads in this block are out of bounds
    if base_pair_idx >= total_pairs:
        return

    # Compute indices from pair_idx
    p = pair_offsets % num_pairs

    temp = pair_offsets // num_pairs
    h = temp % num_heads

    temp = temp // num_heads
    s = temp % seq_len
    b = temp // seq_len

    # Mask for valid indices
    mask = pair_offsets < total_pairs

    # Compute offsets for x
    offset_x = b * stride_batch + s * stride_seq + h * stride_head + 2 * p
    offset_x1 = offset_x + 1

    # Compute offsets for cos and sin
    offset_theta = b * stride_batch_theta + s * stride_seq_theta + p

    # Load data with bounds checking
    x0 = tl.load(x_ptr + offset_x, mask=mask, other=0.0)
    x1 = tl.load(x_ptr + offset_x1, mask=mask, other=0.0)
    cos_th = tl.load(cos_ptr + offset_theta, mask=mask, other=0.0)
    sin_th = tl.load(sin_ptr + offset_theta, mask=mask, other=0.0)

    # Compute rotated values
    x0_new = x0 * cos_th - x1 * sin_th
    x1_new = x0 * sin_th + x1 * cos_th
    tl.store(x_ptr + offset_x, x0_new, mask=mask)
    tl.store(x_ptr + offset_x1, x1_new, mask=mask)


def rope_triton(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Ensure it is complex
    if not freqs_cis.is_complex():
        freqs_cis = torch.view_as_complex(freqs_cis)

    # Extract cos and sin from freqs_cis
    cos_theta = freqs_cis.real.contiguous().half()
    sin_theta = freqs_cis.imag.contiguous().half()

    batch_size, seq_len, num_heads_q, head_dim = xq.shape
    _, _, num_heads_k, _ = xk.shape
    num_pairs = head_dim // 2

    # Process xq
    total_elements_q = batch_size * seq_len * num_heads_q * num_pairs

    def grid_q(META):
        return (triton.cdiv(total_elements_q, META["BLOCK_SIZE"]),)

    rope_kernel_inplace[grid_q](
        xq,
        cos_theta,
        sin_theta,
        batch_size,
        seq_len,
        num_heads_q,
        *xq.stride()[:3],
        *cos_theta.stride()[:2],
        num_pairs,
    )

    # Process xk
    total_elements_k = batch_size * seq_len * num_heads_k * num_pairs

    def grid_k(META):
        return (triton.cdiv(total_elements_k, META["BLOCK_SIZE"]),)

    rope_kernel_inplace[grid_k](
        xk,
        cos_theta,
        sin_theta,
        batch_size,
        seq_len,
        num_heads_k,
        *xk.stride()[:3],
        *cos_theta.stride()[:2],
        num_pairs,
    )

    return xq, xk
