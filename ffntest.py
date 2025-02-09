import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@torch.no_grad()
def ffn_forward(
    x: torch.Tensor,
    dim: int,
    hidden_dim: int,
    w1: torch.nn.Linear,
    w2: torch.nn.Linear,
    w3: torch.nn.Linear,
) -> torch.Tensor:
    return w2(F.silu(w1(x)) * w3(x))


def get_configs():
    configs = []
    for k in range(8, 11):
        configs.append(
            triton.Config(
                {
                    "BLOCK_D": 512,
                    "BLOCK_K": 2**k,
                },
                num_warps=4,
                num_stages=3,
            )
        )
    return configs


@triton.autotune(configs=get_configs(), key=["hidden_dim"])
@triton.jit
def ffn_forward_kernel(
    x_ptr,
    w1_ptr,
    w3_ptr,
    w2_t_ptr,
    output_ptr,
    input_dim: tl.constexpr,
    hidden_dim: tl.constexpr,
    seq_len: tl.constexpr,
    output_dim: tl.constexpr,
    x_batch_stride,
    x_seq_stride,
    x_feature_stride,
    w1_row_stride,
    w1_feature_stride,
    w3_row_stride,
    w3_feature_stride,
    w2_t_row_stride,
    output_batch_stride,
    output_seq_stride,
    output_feature_stride,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_ij = tl.program_id(0)
    pid_k = tl.program_id(1)
    i = pid_ij // seq_len
    j = pid_ij % seq_len

    x_offset = i * x_batch_stride + j * x_seq_stride

    # Current k block indices
    k_indices = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_indices < output_dim

    # Initialize accumulator
    output_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # Loop over all hidden_dim (m)
    for m in range(hidden_dim):
        w1x = 0.0
        w3x = 0.0
        # Loop over input_dim in blocks
        for d_block in range(0, tl.cdiv(input_dim, BLOCK_D)):
            d_indices = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
            d_mask = d_indices < input_dim

            x_ptrs = x_ptr + x_offset + d_indices * x_feature_stride
            x = tl.load(x_ptrs, mask=d_mask, other=0.0)

            w1_ptrs = w1_ptr + m * w1_row_stride + d_indices * w1_feature_stride
            w1 = tl.load(w1_ptrs, mask=d_mask, other=0.0)
            w1x += tl.sum(x * w1)

            w3_ptrs = w3_ptr + m * w3_row_stride + d_indices * w3_feature_stride
            w3 = tl.load(w3_ptrs, mask=d_mask, other=0.0)
            w3x += tl.sum(x * w3)

        # Compute a = SiLU(w1x) * w3x
        silu = tl.sigmoid(w1x) * w1x
        a = silu * w3x

        # Load w2_t[m, k] for current k block
        w2_t_ptrs = w2_t_ptr + m * w2_t_row_stride + k_indices
        w2_t = tl.load(w2_t_ptrs, mask=k_mask, other=0.0)

        # Accumulate into output
        output_acc += a * w2_t

    # Write output
    output_offset = (
        i * output_batch_stride
        + j * output_seq_stride
        + k_indices * output_feature_stride
    )
    output_ptrs = output_ptr + output_offset
    tl.store(output_ptrs, output_acc, mask=k_mask)


def ffn_forward_triton(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, input_dim = x.shape
    hidden_dim, _ = w1.shape
    output_dim, _ = w2.shape
    print(hidden_dim, output_dim, input_dim)
    output = torch.empty(
        (batch_size, seq_len, output_dim), device=x.device, dtype=x.dtype
    )

    def grid(META):
        return (batch_size * seq_len, triton.cdiv(output_dim, META["BLOCK_K"]))

    w2_t = w2.T.contiguous()

    ffn_forward_kernel[grid](
        x,
        w1,
        w3,
        w2_t,
        output,
        input_dim,
        hidden_dim,
        seq_len,
        output_dim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w1.stride(0),
        w1.stride(1),
        w3.stride(0),
        w3.stride(1),
        w2_t.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
    )
    return output


def test_ffn():
    torch.manual_seed(0)
    x = torch.randn(2, 256, 2304, device="cuda", dtype=torch.float16)
    dim = 2304
    hidden_dim = 9216
    w1 = torch.nn.Linear(
        dim, hidden_dim, bias=False, device="cuda", dtype=torch.float16
    )
    w2 = torch.nn.Linear(
        hidden_dim, dim, bias=False, device="cuda", dtype=torch.float16
    )
    w3 = torch.nn.Linear(
        dim, hidden_dim, bias=False, device="cuda", dtype=torch.float16
    )

    out_torch = ffn_forward(x, dim, hidden_dim, w1, w2, w3)
    out_triton = ffn_forward_triton(x, w1.weight, w2.weight, w3.weight)

    print(f"Output max difference: {torch.max(torch.abs(out_torch - out_triton))}")
    torch.testing.assert_close(out_torch, out_triton, atol=1e-4, rtol=1e-4)
    print("Test passed!")

    from time import time

    # Benchmarking
    n_repeat = 100
    print("Benchmarking...")
    t0 = time()
    for _ in range(n_repeat):
        out_triton = ffn_forward_triton(x, w1.weight, w2.weight, w3.weight)
    triton_time = time() - t0
    t0 = time()
    for _ in range(n_repeat):
        out_triton = ffn_forward(x, 0, 0, w1, w2, w3)
    torch_time = time() - t0
    print(f"Triton time: {triton_time:.5f}ms")
    print(f"Torch time: {torch_time:.5f}ms")


if __name__ == "__main__":
    test_ffn()
