import torch
import triton
import triton.language as tl

# from triton.language.extra import libdevice


def configs():
    configs = []
    for h in [4, 8]:
        configs.append(
            triton.Config(
                {"BLOCK_SIZE_H": h, "BLOCK_SIZE_R": 256},
                num_stages=2,
                num_warps=1,
            )
        )
    return configs


# fmt: off
@triton.autotune(configs(), key=["HD"])
@triton.jit
def softmax_kernel(
    x_ptr,
    xb_stride, xc_stride, xh_stride, xr_stride,
    HD: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    # fmt: on
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    h_idx = tl.program_id(2)

    h_start = h_idx * BLOCK_SIZE_H
    h_offset = tl.arange(0, BLOCK_SIZE_H)
    current_h = h_start + h_offset
    valid_h = current_h < HD

    base_ptrs = (
        x_ptr
        + b_idx * xb_stride
        + c_idx * xc_stride
        + current_h * xh_stride
    )

    row_max = tl.full((BLOCK_SIZE_H,), -float("inf"), dtype=tl.float32)
    row_sum = tl.full((BLOCK_SIZE_H,), 0.0, dtype=tl.float32)

    for offset in range(0, HD, BLOCK_SIZE_R):
        cols = offset + tl.arange(0, BLOCK_SIZE_R)
        mask = cols < HD

        ptrs = base_ptrs[:, None] + cols[None, :] * xr_stride
        x = tl.load(ptrs, mask=valid_h[:, None] & mask[None, :], other=-float("inf"))

        row_max = tl.maximum(row_max, tl.max(x, axis=1))

    for offset in range(0, HD, BLOCK_SIZE_R):
        cols = offset + tl.arange(0, BLOCK_SIZE_R)
        mask = cols < HD

        ptrs = base_ptrs[:, None] + cols[None, :] * xr_stride
        x = tl.load(ptrs, mask=valid_h[:, None] & mask[None, :], other=0.0)

        exp_x = tl.exp(x - row_max[:, None])
        row_sum += tl.sum(exp_x, axis=1)

    for offset in range(0, HD, BLOCK_SIZE_R):
        cols = offset + tl.arange(0, BLOCK_SIZE_R)
        mask = cols < HD

        ptrs = base_ptrs[:, None] + cols[None, :] * xr_stride
        x = tl.load(ptrs, mask=valid_h[:, None] & mask[None, :], other=0.0)

        exp_x = tl.exp(x - row_max[:, None])
        normalized = exp_x / row_sum[:, None]
        tl.store(ptrs, normalized.to(tl.bfloat16), mask=valid_h[:, None] & mask[None, :])


def inplace_tiled_softmax(x: torch.Tensor):
    if not x.is_contiguous():
        x = x.contiguous()

    def grid(META):
        return (*x.shape[:2], triton.cdiv(x.shape[2], META["BLOCK_SIZE_H"]))

    softmax_kernel[grid](x, *x.stride(), x.shape[3])

    return x
