import torch
import triton
import triton.language as tl


@triton.jit
def _triton_merged_rotary_embedding(
    xq_ptr,
    xk_ptr,
    freqs_cos_ptr,
    freqs_sin_ptr,
    xq_row_stride,
    xk_row_stride,
    freqs_row_stride,
    seq_len,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # locate start address
    xq_ptr = xq_ptr + pid * xq_row_stride
    xk_ptr = xk_ptr + pid * xk_row_stride

    # Get the batch index and sequence index
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len

    # Load the cosine and sine values for the current token position
    freqs_cos = (
        freqs_cos_ptr
        + batch_idx * (seq_len * freqs_row_stride)
        + seq_idx * freqs_row_stride
    )
    freqs_sin = (
        freqs_sin_ptr
        + batch_idx * (seq_len * freqs_row_stride)
        + seq_idx * freqs_row_stride
    )

    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(freqs_cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(freqs_sin + cos_offsets, mask=cos_mask, other=0)

    # Load the left and right half of xq and xk for the current program instance
    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_half_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )
    xq_tile_1 = tl.load(xq_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(
        sin_row.dtype
    )
    xk_tile_1 = tl.load(xk_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(
        sin_row.dtype
    )

    # right half of the head
    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    xq_tile_2 = tl.load(xq_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(
        sin_row.dtype
    )
    xk_tile_2 = tl.load(xk_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(
        sin_row.dtype
    )

    # Apply the rotary embedding
    new_xq_tile_1 = xq_tile_1 * cos_row - xq_tile_2 * sin_row
    tl.store(xq_ptr + first_half_q_offsets, new_xq_tile_1, mask=first_q_mask)
    new_xq_tile_2 = xq_tile_2 * cos_row + xq_tile_1 * sin_row
    tl.store(xq_ptr + second_half_q_offsets, new_xq_tile_2, mask=second_q_mask)

    new_xk_tile_1 = xk_tile_1 * cos_row - xk_tile_2 * sin_row
    tl.store(xk_ptr + first_half_k_offsets, new_xk_tile_1, mask=first_k_mask)
    new_xk_tile_2 = xk_tile_2 * cos_row + xk_tile_1 * sin_row
    tl.store(xk_ptr + second_half_k_offsets, new_xk_tile_2, mask=second_k_mask)


def apply_merged_rotary_embedding(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs_cos, freqs_sin = torch.view_as_real(freqs_cis).split(1, dim=-1)

    # Transpose xq and xk to match the expected layout
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = xq.shape
    n_kv_head = xk.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

    n_row = batch_size * seq_len

    # Ensure tensors passed into the kernel are contiguous
    xq = xq.contiguous()
    xk = xk.contiguous()
    freqs_cos = freqs_cos.contiguous()
    freqs_sin = freqs_sin.contiguous()

    _triton_merged_rotary_embedding[(n_row,)](
        xq,
        xk,
        freqs_cos,
        freqs_sin,
        xq.stride(1),
        xk.stride(1),
        freqs_cos.stride(-2),
        seq_len,
        batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Transpose back to the original shape
    return xq.transpose(1, 2), xk.transpose(1, 2)
