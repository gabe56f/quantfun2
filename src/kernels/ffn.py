import torch
import triton
import triton.language as tl


@triton.jit
def fused_ffn_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    w3_ptr,
    output_ptr,
    n_elements: tl.constexpr,  # sequence length * batch size
    d_in: tl.constexpr,
    d_hidden: tl.constexpr,
    d_out: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_DOUT: tl.constexpr,
    BLOCK_SIZE_DHIDDEN: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start_n = pid * BLOCK_SIZE_N
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < n_elements

    block_start_dout = tl.program_id(1) * BLOCK_SIZE_DOUT
    offsets_dout = block_start_dout + tl.arange(0, BLOCK_SIZE_DOUT)
    mask_dout = offsets_dout < d_out

    # --- x ---
    x_offsets = offsets_n[:, None] * d_in + tl.arange(0, d_in)[None, :]
    x_ptrs = x_ptr + x_offsets
    x_block = tl.load(x_ptrs, mask=mask_n[:, None], other=0.0)

    # --- output ---
    output_block = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_DOUT), dtype=tl.float32)

    for start_dh in range(0, d_hidden, BLOCK_SIZE_DHIDDEN):
        offsets_dh = start_dh + tl.arange(0, BLOCK_SIZE_DHIDDEN)
        mask_dh = offsets_dh < d_hidden

        # --- w1 ---
        w1_offsets = offsets_dh[:, None] * d_in + tl.arange(0, d_in)[None, :]
        w1_ptrs = w1_ptr + w1_offsets
        w1_block = tl.load(w1_ptrs, mask=mask_dh[:, None], other=0.0)

        # --- w3 ---
        w3_offsets = offsets_dh[:, None] * d_in + tl.arange(0, d_in)[None, :]
        w3_ptrs = w3_ptr + w3_offsets
        w3_block = tl.load(w3_ptrs, mask=mask_dh[:, None], other=0.0)

        # --- w2 ---
        w2_offsets = offsets_dout[None, :] * d_hidden + offsets_dh[:, None]
        w2_ptrs = w2_ptr + w2_offsets
        w2_block = tl.load(
            w2_ptrs, mask=mask_dout[None, :] & mask_dh[:, None], other=0.0
        )

        # --- silu(w1 @ x) * (w3 @ x) ---
        w1x = tl.sum(w1_block * x_block[None, :], axis=1)  # [BLOCK_SIZE_DHIDDEN]
        w3x = tl.sum(w3_block * x_block[None, :], axis=1)  # [BLOCK_SIZE_DHIDDEN]
        silu_w1x_w3x = tl.sigmoid(w1x) * w1x * w3x  # [BLOCK_SIZE_DHIDDEN]

        # --- w2 @ (silu(w1 @ x) * (w3 @ x)) ---
        output_block += tl.dot(
            silu_w1x_w3x[:, None], w2_block, allow_tf32=False
        )  # [BLOCK_SIZE_DHIDDEN, BLOCK_SIZE_DOUT] -> [BLOCK_SIZE_DOUT] and then broadcasted to [BLOCK_SIZE_N, BLOCK_SIZE_DOUT] but actually it should be [BLOCK_SIZE_N, BLOCK_SIZE_DOUT] from the start. Let's fix the shapes

    output_offsets = offsets_n[:, None] * d_out + offsets_dout[None, :]
    output_ptrs = output_ptr + output_offsets
    tl.store(output_ptrs, output_block, mask=mask_n[:, None] & mask_dout[None, :])


def dequant_if_need(x: torch.nn.Linear) -> torch.Tensor:
    if hasattr(x.weight, "dequantize"):
        return x.weight.dequantize().T
    return x.weight.T


def ffn_forward(
    x: torch.Tensor,
    dim: int,
    hidden_dim: int,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> torch.Tensor:
    w1 = dequant_if_need(w1)
    w2 = dequant_if_need(w2)
    w3 = dequant_if_need(w3)

    output = torch.empty_like(x)
    batch_size, seq_len, dim_in = x.shape
    n_elements = batch_size * seq_len

    BLOCK_SIZE_N = 128
    BLOCK_SIZE_DOUT = 32
    BLOCK_SIZE_DHIDDEN = 128
    grid = (
        triton.cdiv(n_elements, BLOCK_SIZE_N),
        triton.cdiv(dim, BLOCK_SIZE_DOUT),
    )
    num_warps = 4
    num_stages = 1

    fused_ffn_kernel[grid](
        x_ptr=x.data_ptr(),
        w1_ptr=w1.data_ptr(),
        w2_ptr=w2.data_ptr(),
        w3_ptr=w3.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=n_elements,
        d_in=dim,
        d_hidden=hidden_dim,
        d_out=dim,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_DOUT=BLOCK_SIZE_DOUT,
        BLOCK_SIZE_DHIDDEN=BLOCK_SIZE_DHIDDEN,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
