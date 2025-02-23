import triton
import triton.language as tl
import torch


def _configs():
    configs = []
    for m, n in zip(range(6, 8), range(5, 7)):
        for g in range(2, 5):
            for s in range[3, 4, 7]:
                for w in [2, 4]:
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_SIZE_M": 2**m,
                                "BLOCK_SIZE_N": 2**n,
                                "BLOCK_SIZE_K": 2**n,
                                "GROUP_SIZE_M": 2**g,
                            },
                            num_stages=s,
                            num_warps=w,
                        )
                    )

    return configs


@triton.autotune(configs=_configs(), key=["M"])
@triton.jit
def _triton_matmul_index_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    idx_ptr,
    B,
    M,
    N,
    K,
    stride_az,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cz,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (N, K) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(B * M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % (B * M)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (
        a_ptr
        + (offs_am[:, None] // M) * stride_az
        + (offs_am[:, None] % M) * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_cm[:, None] < B * M) & (offs_cn[None, :] < N)
    offs_cm = tl.load(idx_ptr + offs_am % M)
    offs_cz = offs_am // M
    c_ptrs = (
        c_ptr
        + offs_cz[:, None] * stride_cz
        + offs_cm[:, None] * stride_cm
        + offs_cn[None, :] * stride_cn
    )
    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=c_mask)


def _partially_linear(
    x: torch.Tensor,  # [BATCH, N_CTX, C_IN]
    w: torch.Tensor,  # [C_OUT, C_IN]
    index: torch.Tensor,  # [N_CTX, ]
) -> torch.Tensor:  # [BATCH, N_CTX, C_OUT]
    B, M, K = x.shape
    N = w.shape[0]

    def grid(META):
        return (
            triton.cdiv(B * M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    outputs = torch.empty_like(x)

    # fmt: off
    _triton_matmul_index_kernel[grid](
        x, w, outputs, index,
        B, M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1),
        outputs.stride(0), outputs.stride(1), outputs.stride(2),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8,
        num_stages=4, num_warps=4,
    )
    # fmt: on

    return outputs
