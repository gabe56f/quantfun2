import torch


def binary_search_topk(v, k, max_iter=None):
    """
    Binary Search-based Top-k Algorithm (with optional Early Stopping).

    Args:
        v: Input vector (torch.Tensor).
        k: The number of largest elements to select.
        max_iter: Maximum number of iterations for early stopping.
                  If None, run until convergence (Algorithm 1).
                  If an integer, implement early stopping (Algorithm 2).

    Returns:
        A tuple containing:
            - elems: The top-k largest elements.
            - indices: The corresponding indices of the top-k elements in v.
    """
    min_val = torch.min(v)
    max_val = torch.max(v)

    if max_iter is None:  # Algorithm 1: No Early Stopping
        epsilon = 1e-4 * max_val
        while max_val - min_val > epsilon:
            mid = (min_val + max_val) / 2
            cnt = torch.sum((v >= mid).int())  # Count elements >= mid

            if cnt == k:
                elems = v[v >= mid]
                indices = torch.nonzero(
                    v >= mid
                ).squeeze()  # Use nonzero to get indices
                return elems, indices
            elif cnt < k:
                max_val = mid
            else:  # cnt > k
                min_val = mid

        # Handle the case where the loop exits due to epsilon (cnt != k)
        elems_ge_mid_plus_eps = v[v >= mid + epsilon]  # using mid + eps to get elements
        indices_ge_mid_plus_eps = torch.nonzero(v >= mid + epsilon).squeeze()

        elems_between = v[
            (v >= mid - epsilon) & (v < mid + epsilon)
        ]  # between [mid-eps,mid+eps)
        indices_between = torch.nonzero(
            (v >= mid - epsilon) & (v < mid + epsilon)
        ).squeeze()

        # Concatenate and take the first k elements
        num_remaining = k - elems_ge_mid_plus_eps.numel()
        elems = torch.cat([elems_ge_mid_plus_eps, elems_between[:num_remaining]])
        indices = torch.cat([indices_ge_mid_plus_eps, indices_between[:num_remaining]])
        return elems, indices

    else:  # Algorithm 2: With Early Stopping
        for _ in range(max_iter):
            mid = (min_val + max_val) / 2
            cnt = torch.sum((v >= mid).int())

            if cnt == k:
                elems = v[v >= mid]
                indices = torch.nonzero(v >= mid).squeeze()
                return elems, indices

            elif cnt < k:
                max_val = mid
            else:
                min_val = mid

        # After max_iter iterations, use min_val as the threshold
        elems = v[v >= min_val]
        indices = torch.nonzero(v >= min_val).squeeze()
        # Take the first k
        return elems[:k], indices[:k]


def row_wise_topk(matrix, k, max_iter=None):
    """
    Applies the binary search top-k algorithm row-wise to a matrix.

    Args:
        matrix: Input matrix (torch.Tensor).
        k: The number of largest elements to select per row.
        max_iter:  Maximum iterations for early stopping (optional).

    Returns:
        A tuple containing:
            - elems: Top-k elements for each row (list of tensors).
            - indices: Corresponding indices for each row (list of tensors).
    """
    all_elems = []
    all_indices = []
    for row in matrix:
        elems, indices = binary_search_topk(row, k, max_iter)
        all_elems.append(elems)
        all_indices.append(indices)
    return all_elems, all_indices


def test():
    # --- Test Cases ---
    # Test case 1: Basic test with a small vector (Algorithm 1)
    v1 = torch.tensor([1, 5, 2, 8, 3, 9, 4, 7, 6])
    k1 = 3
    elems1, indices1 = binary_search_topk(v1, k1)
    print(f"Test Case 1 (No Early Stopping): elems={elems1}, indices={indices1}")
    expected_elems1 = torch.tensor([9, 8, 7])  # Example expected values (may vary)
    expected_indices1 = torch.tensor([5, 3, 7])
    expected_elems1, _ = torch.sort(expected_elems1, descending=True)
    elems1, _ = torch.sort(elems1, descending=True)
    assert torch.allclose(
        elems1, expected_elems1
    ), f"Test Case 1 Failed: {elems1} != {expected_elems1}"
    assert torch.all(
        torch.isin(expected_indices1, indices1)
    ), f"Indices mismatch: {indices1}"

    # Test case 2: Basic test with early stopping
    v2 = torch.tensor([1, 5, 2, 8, 3, 9, 4, 7, 6])
    k2 = 3
    max_iter2 = 5  # Early stopping after 5 iterations
    elems2, indices2 = binary_search_topk(v2, k2, max_iter2)
    print(f"Test Case 2 (Early Stopping): elems={elems2}, indices={indices2}")

    # Test case 3: Row-wise top-k on a matrix
    matrix3 = torch.tensor([[1, 5, 2], [8, 3, 9], [4, 7, 6]])
    k3 = 2
    elems3, indices3 = row_wise_topk(matrix3, k3)
    print(
        f"Test Case 3 (Row-wise, No Early Stopping): elems={elems3}, indices={indices3}"
    )

    # Test case 4: Row-wise top-k with early stopping
    matrix4 = torch.tensor([[1, 5, 2], [8, 3, 9], [4, 7, 6]])
    k4 = 2
    max_iter4 = 3  # Early stopping after 3 iterations
    elems4, indices4 = row_wise_topk(matrix4, k4, max_iter4)
    print(f"Test Case 4 (Row-wise, Early Stopping): elems={elems4}, indices={indices4}")

    # Test case 5: Larger matrix and k (for performance and correctness)
    matrix5 = torch.randn(100, 256)  # 100 rows, 256 columns
    k5 = 64
    elems5, indices5 = row_wise_topk(matrix5, k5)
    print(
        f"Test Case 5 (Large Matrix): elems (first row)={elems5[0]}, indices (first row)={indices5[0]}"
    )

    # Test case 6:  Larger Matrix, No Early Stopping (to test edge cases of Algorithm 1)
    matrix6 = torch.randn(100, 256)
    k6 = 16
    elems6, indices6 = row_wise_topk(matrix6, k6)
    print(
        f"Test case 6 (Large Matrix, No Early Stopping): elems (first row)={elems6[0]}, indices(first row)={indices6[0]}"
    )

    # Test case 7: Vector with many duplicate values
    v7 = torch.tensor([5, 5, 5, 2, 2, 8, 8, 8, 1, 9, 9])
    k7 = 4
    elems7, indices7 = binary_search_topk(v7, k7)
    print(f"Test case 7 (duplicates): elems={elems7}, indices={indices7}")
    # Test Case 8:  Edge Case, k = 1
    v8 = torch.randn(100)
    k8 = 1
    elems8, indices8 = binary_search_topk(v8, k8)
    print(f"Test Case 8 (k=1): elems={elems8}, indices={indices8}")
    assert elems8.numel() == 1 and indices8.numel() == 1

    # Test Case 9: Edge Case, k = M (length of vector)
    v9 = torch.randn(100)
    k9 = v9.numel()
    elems9, indices9 = binary_search_topk(v9, k9)
    print(
        f"Test Case 9 (k=M): elems (first 5)={elems9[:5]}, indices (first 5)={indices9[:5]}"
    )
    assert elems9.numel() == k9 and indices9.numel() == k9
    assert torch.allclose(torch.sort(v9, descending=True)[0], elems9)

    # Test Case 10: Edge Case, All elements are equal
    v10 = torch.ones(100)
    k10 = 10
    elems10, indices10 = binary_search_topk(v10, k10)
    print(f"Test Case 10 (all equal): elems={elems10}, indices={indices10}")
    assert elems10.numel() == k10 and indices10.numel() == k10

    print("All test cases run.")


if __name__ == "__main__":
    test()
