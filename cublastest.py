# import torch
# from triton._C.libtriton import nvidia

# cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
# cublas = nvidia.cublas.CublasLt(cublas_workspace)

# a = torch.randn(4096, 4096, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
# b = torch.randn(4096, 4096, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
# M, k = a.shape
# N, k = b.shape

# c = torch.empty(M, N, dtype=torch.float8_e4m3fn)

# cublas.matmul(a, b, c)
# # print(c.sum())
# # print(torch.matmul(a, b).sum())
# tc = torch.matmul(a, b)
# torch.testing.assert_close(c, tc, rtol=1e-3, atol=1e-3)
