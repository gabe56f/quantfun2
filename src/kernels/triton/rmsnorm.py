import torch
import triton
import triton.language as tl


def get_configs():
    configs = []
    for k in range(6, 11):
        for warp in [1, 2, 8]:
            configs.append(
                triton.Config(
                    {
                        "BLOCK_SIZE": 2**k,
                    },
                    num_warps=warp,
                    num_stages=1,
                )
            )
    return configs


@triton.autotune(configs=get_configs(), key=["N"], warmup=10, rep=30)
@triton.jit
def rms_norm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    stride_x_row,
    stride_x_col,
    stride_weight_col,
    stride_output_row,
    stride_output_col,
    eps: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    x_row_ptr = x_ptr + row_idx * stride_x_row
    output_row_ptr = output_ptr + row_idx * stride_output_row

    sum_squares = 0.0
    for col_offset in range(0, N, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_val = tl.load(x_row_ptr + cols * stride_x_col, mask=mask, other=0.0)
        x_val_float = x_val.to(tl.float32)
        sum_squares += tl.sum(x_val_float * x_val_float, axis=0)

    mean = sum_squares / N + eps
    rsqrt_val = tl.rsqrt(mean)

    for col_offset in range(0, N, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_val = tl.load(x_row_ptr + cols * stride_x_col, mask=mask)
        weight_val = tl.load(weight_ptr + cols * stride_weight_col, mask=mask)

        x_val_float = x_val.to(tl.float32)
        weight_val_float = weight_val.to(tl.float32)

        scaled_x = x_val_float * rsqrt_val
        output_val_float = scaled_x * weight_val_float

        output_val = output_val_float.to(x_val.dtype)
        tl.store(output_row_ptr + cols * stride_output_col, output_val, mask=mask)


def apply_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    output = torch.empty_like(x)
    dim = weight.shape[0]

    x_2d = x.contiguous().view(-1, dim)
    M, N = x_2d.shape

    weight_1d = weight.contiguous().view(-1)

    grid = (M,)
    rms_norm_kernel[grid](
        x_2d,
        weight_1d,
        output.view(-1, dim),
        x_2d.stride(0),
        x_2d.stride(1),
        weight_1d.stride(0),
        output.view(-1, dim).stride(0),
        output.view(-1, dim).stride(1),
        eps=eps,
        M=M,
        N=N,
    )
    return output
