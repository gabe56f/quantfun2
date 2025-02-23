#include <cuda_bf16.h>

extern "C"
{
    __device__ float2 complex_multiplication(float2 a, float2 b)
    {
        float2 res;
        res.x = a.x * b.x - a.y * b.y;
        res.y = a.x * b.y + a.y * b.x;
        return res;
    }
    __global__ void PYCUDA_ROPE_KERNEL(
        short *x,       // [B, N, H, D]
        float *f,       // [1, N, D]
        long long *idx, // [N]
        short *y,       // [B, M, H, D]
        const int M,
        const int N,
        const int H,
        const int D)
    {
        const int n_idx = blockIdx.x * blockDim.y + threadIdx.y;
        if (n_idx >= N)
            return;
        int m_idx = idx[n_idx];
        x += blockIdx.y * N * H * D + n_idx * H * D + threadIdx.x * D;
        y += blockIdx.y * M * H * D + m_idx * H * D + threadIdx.x * D;
        f += n_idx * D;
        float4 buf_a;
        float4 buf_b;
        float4 buf_res;
        float2 complex_a;
        float2 complex_b;
        __nv_bfloat162 complex_res;
        // TODO: unroll
        for (int offset = 0; offset < D; offset += 8)
        {
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                if (i % 4 == 0)
                    buf_a = (reinterpret_cast<float4 *>(&x[offset]))[0];
                if (i % 2 == 0)
                    buf_b = (reinterpret_cast<float4 *>(&f[offset + i * 2]))[0];
                complex_a = __bfloat1622float2((reinterpret_cast<__nv_bfloat162 *>(&buf_a))[i]);
                complex_b = (reinterpret_cast<float2 *>(&buf_b))[i % 2];
                complex_res = __float22bfloat162_rn(complex_multiplication(complex_a, complex_b));
                (reinterpret_cast<float *>(&buf_res))[i] = (reinterpret_cast<float *>(&complex_res))[0];
            }
            (reinterpret_cast<float4 *>(&y[offset]))[0] = buf_res;
        }
    }
}