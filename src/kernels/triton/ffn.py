import torch
import triton
import triton.language as tl


def _configs():
    configs = []
    for BLOCK_SIZE in [128, 512, 1024]:
        for num_warps in [2, 4, 16]:
            configs.append(
                triton.Config(
                    {"BLOCK_SIZE": BLOCK_SIZE},
                    num_warps=num_warps,
                    num_stages=1,
                )
            )
    return configs


@triton.autotune(configs=_configs(), key=["hidden_size"], rep=50, warmup=20)
@triton.jit
def modulated_ln_kernel_inplace(
    x_ptr,
    scale_ptr,
    batch_size,
    num_patches,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // num_patches
    p = pid % num_patches
    if b >= batch_size or p >= num_patches:
        return

    x_offset = b * num_patches * hidden_size + p * hidden_size
    scale_offset = b * hidden_size

    h_range = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + x_offset + h_range, mask=h_range < hidden_size, other=0.0)

    # Compute mean
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / hidden_size

    # Compute variance
    diff = x - mean
    diff2 = diff * diff
    sum_diff2 = tl.sum(diff2, axis=0)
    var = sum_diff2 / hidden_size
    denom = tl.sqrt(var + eps)

    scale = tl.load(
        scale_ptr + scale_offset + h_range, mask=h_range < hidden_size, other=0.0
    )

    ln_x = (x - mean) / denom
    modulated_ln_x = ln_x * (1.0 + scale)

    tl.store(x_ptr + x_offset + h_range, modulated_ln_x, mask=h_range < hidden_size)


# @torch.compile()
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
    batch_size = x.shape[0]
    grid = (batch_size * num_patches,)
    modulated_ln_kernel_inplace[grid](
        x,
        scale,
        batch_size,
        num_patches,
        hidden_size,
        1e-6,
    )

    x_reshaped = x.view(batch_size, -1, hidden_size)
    output = torch.nn.functional.linear(x_reshaped, linear_weight, linear_bias)

    return output.view(batch_size, -1, num_patches * out_channels)
