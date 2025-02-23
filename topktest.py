import torch
import triton
import triton.language as tl


@triton.jit
def _rtopk_kernel(
    v_ptr,
    output_elements_ptr,
    output_indices_ptr,
    stride_v,
    stride_oe,
    stride_oi,
    k,
    max_iter,
    BLOCK_SIZE: tl.constexpr,
    num_elements: tl.constexpr,
):
    """
    Triton kernel for the rtopk function.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Load the block of data
    v = tl.load(v_ptr + offsets * stride_v, mask=mask, other=-float("inf"))

    # Find min and max (reduce within the block)
    min_val = tl.min(v, axis=0)
    max_val = tl.max(v, axis=0)

    for _ in range(max_iter):
        mid = (min_val + max_val) / 2
        # Count elements >= mid (reduce within the block)
        cnt = tl.sum((v >= mid).to(tl.int32), axis=0)

        if cnt == k:
            # Create a mask for elements >= mid
            ge_mid_mask = (v >= mid) & mask
            num_ge_mid = tl.sum(ge_mid_mask.to(tl.int32))

            # Gather elements and indices >= mid
            elements = tl.where(ge_mid_mask, v, 0)  # Replace non-matching with 0
            indices = tl.where(ge_mid_mask, offsets, 0)

            # Sort within the block
            _, sorted_indices_in_block = tl.sort(elements, descending=True)

            # Manual gather (reorder)
            sorted_elements = tl.zeros_like(elements)
            sorted_indices = tl.zeros_like(indices)
            for i in range(BLOCK_SIZE):  # Iterate through sorted indices
                idx = sorted_indices_in_block[i]
                valid_idx = i < num_ge_mid  # Check if index is within the valid range
                sorted_elements = tl.where(
                    valid_idx,
                    tl.load(
                        v_ptr + (block_start + idx) * stride_v,
                        mask=(block_start + idx) < num_elements,
                        other=0,
                    ),
                    sorted_elements,
                )  # load with offset
                sorted_indices = tl.where(
                    valid_idx,
                    tl.load(
                        output_indices_ptr + (block_start + idx) * stride_oi,
                        mask=(block_start + idx) < num_elements,
                        other=0,
                    )
                    + idx,
                    sorted_indices,
                )  # load old val, and add the sorted index

            # Store up to k results
            results_mask = tl.arange(0, BLOCK_SIZE) < k
            tl.store(
                output_elements_ptr
                + (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * stride_oe,
                sorted_elements,
                mask=results_mask & (tl.arange(0, BLOCK_SIZE) < num_ge_mid),
            )
            tl.store(
                output_indices_ptr
                + (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * stride_oi,
                sorted_indices,
                mask=results_mask & (tl.arange(0, BLOCK_SIZE) < num_ge_mid),
            )
            return

        elif cnt < k:
            max_val = mid
        else:
            min_val = mid

    # Final iteration (handle cases where cnt != k within max_iter)
    ge_min_mask = (v >= min_val) & mask
    num_ge_min = tl.sum(ge_min_mask.to(tl.int32))

    # Gather elements and indices >= min_val
    elements = tl.where(ge_min_mask, v, 0)
    indices = tl.where(ge_min_mask, offsets, 0)

    # Sort within the block
    _, sorted_indices_in_block = tl.sort(elements, descending=True)

    # Manual gather (reorder)
    sorted_elements = tl.zeros_like(elements)
    sorted_indices = tl.zeros_like(indices)
    for i in range(BLOCK_SIZE):  # Iterate through sorted indices
        idx = sorted_indices_in_block[i]
        valid_idx = i < num_ge_min  # check for num_ge_min

        sorted_elements = tl.where(
            valid_idx,
            tl.load(
                v_ptr + (block_start + idx) * stride_v,
                mask=(block_start + idx) < num_elements,
                other=0,
            ),
            sorted_elements,
        )
        sorted_indices = tl.where(
            valid_idx,
            tl.load(
                output_indices_ptr + (block_start + idx) * stride_oi,
                mask=(block_start + idx) < num_elements,
                other=0,
            )
            + idx,
            sorted_indices,
        )  # load old, add current

    # Store up to k results
    results_mask = tl.arange(0, BLOCK_SIZE) < k
    tl.store(
        output_elements_ptr + (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * stride_oe,
        sorted_elements,
        mask=results_mask & (tl.arange(0, BLOCK_SIZE) < num_ge_min),
    )
    tl.store(
        output_indices_ptr + (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * stride_oi,
        sorted_indices,
        mask=results_mask & (tl.arange(0, BLOCK_SIZE) < num_ge_min),
    )


def rtopk(v: torch.Tensor, k: int, max_iter: int = 6):
    """
    Triton implementation of the rtopk function.

    Args:
        v (torch.Tensor): Input tensor.
        k (int): Number of top elements to return.
        max_iter (int): Maximum number of iterations for binary search.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Top k elements and their indices.
    """
    assert v.is_contiguous(), "Input tensor must be contiguous"
    assert v.ndim == 1, "Input tensor must be 1-dimensional"
    num_elements = v.shape[0]

    # Allocate output tensors
    output_elements = torch.zeros(min(k, num_elements), dtype=v.dtype, device=v.device)
    output_indices = torch.zeros(
        min(k, num_elements), dtype=torch.int64, device=v.device
    )

    if k > num_elements:  # handle cases where k > number of elements
        return v, torch.arange(0, num_elements, device=v.device, dtype=torch.int64)

    # Determine block size (power of 2 for efficiency)
    BLOCK_SIZE = 1024
    if num_elements < BLOCK_SIZE:
        BLOCK_SIZE = triton.next_power_of_2(
            num_elements
        )  # Use a smaller block size if necessary

    # Launch kernel
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)  # Number of blocks
    _rtopk_kernel[grid](
        v,
        output_elements,
        output_indices,
        v.stride(0),
        output_elements.stride(0),
        output_indices.stride(0),
        k,
        max_iter,
        BLOCK_SIZE=BLOCK_SIZE,
        num_elements=num_elements,
    )

    # Final global sort
    sorted_elements, sorted_indices_of_indices = torch.sort(
        output_elements, descending=True
    )
    final_elements = sorted_elements[:k]
    final_indices = output_indices[sorted_indices_of_indices[:k]]

    return final_elements, final_indices


def test_rtopk():
    # Test cases
    test_cases = [
        (torch.randn(10), 3),
        (torch.randn(100), 10),
        (torch.randn(1024), 64),
        (torch.randn(2048), 128),
        (torch.randn(4096), 256),
        (torch.randn(8192), 512),
        (torch.randn(10000), 100),
        (torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), 3),  # Sorted input
        (torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]), 3),  # Reverse sorted input
        (torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]), 3),  # All same values
        (torch.randn(10), 10),  # k == num_elements
        (torch.randn(10), 15),  # k > num_elements,
        (torch.randn(513), 7),  # odd numbers
    ]

    for v, k in test_cases:
        v = v.cuda()
        v_cpu = v.cpu()

        # Triton implementation
        elements_triton, indices_triton = rtopk(v, k)

        # PyTorch implementation (for comparison)
        elements_torch, indices_torch = torch.topk(v_cpu, k)

        # Check results
        assert torch.allclose(
            elements_triton.cpu(), elements_torch, atol=1e-3, rtol=1e-3
        ), f"Elements mismatch for input size {v.shape}, k={k}"
        assert torch.allclose(
            v_cpu[indices_triton.cpu().long()], elements_torch, atol=1e-3
        ), f"Indices-Value mismatch for input size {v.shape}, k={k}"
        print(f"Test passed for input size {v.shape}, k={k}")


if __name__ == "__main__":
    test_rtopk()
    print("All tests passed!")
