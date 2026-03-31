#include "matmul.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <sstream>

namespace
{

    template <int kTile>
    __global__ void matmul_tiled_kernel(const float *a, const float *b, float *c, int m, int n, int k)
    {
        __shared__ float tile_a[kTile][kTile];
        __shared__ float tile_b[kTile][kTile];

        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;

        float acc = 0.0f;
        const int tiles = (k + kTile - 1) / kTile;

        for (int t = 0; t < tiles; ++t)
        {
            const int a_col = t * kTile + threadIdx.x;
            const int b_row = t * kTile + threadIdx.y;

            tile_a[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
            tile_b[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

            __syncthreads();

            for (int x = 0; x < kTile; ++x)
            {
                acc += tile_a[threadIdx.y][x] * tile_b[x][threadIdx.x];
            }

            __syncthreads();
        }

        if (row < m && col < n)
        {
            c[row * n + col] = acc;
        }
    }

    const char *cuda_error_name(cudaError_t code)
    {
        return cudaGetErrorName(code);
    }

    const char *cuda_error_text(cudaError_t code)
    {
        return cudaGetErrorString(code);
    }

    bool check_cuda(cudaError_t code, std::string &error, const char *where)
    {
        if (code == cudaSuccess)
        {
            return true;
        }

        std::ostringstream oss;
        oss << where << " failed: " << cuda_error_name(code) << " (" << cuda_error_text(code) << ")";
        error = oss.str();
        return false;
    }

    template <int kTile>
    bool launch_kernel(const float *d_a,
                       const float *d_b,
                       float *d_c,
                       int m,
                       int n,
                       int k,
                       std::string &error)
    {
        const dim3 block(kTile, kTile);
        const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

        matmul_tiled_kernel<kTile><<<grid, block>>>(d_a, d_b, d_c, m, n, k);

        if (!check_cuda(cudaGetLastError(), error, "kernel launch"))
        {
            return false;
        }

        return check_cuda(cudaDeviceSynchronize(), error, "cudaDeviceSynchronize");
    }

}

bool matmul_gpu(const Matrix &a,
                const Matrix &b,
                Matrix &c,
                int block_size,
                float &elapsed_ms,
                std::string &error)
{
    error.clear();
    elapsed_ms = 0.0f;

    if (a.cols != b.rows)
    {
        error = "matmul_gpu: dimension mismatch";
        return false;
    }

    if (c.rows != a.rows || c.cols != b.cols)
    {
        error = "matmul_gpu: output matrix has invalid shape";
        return false;
    }

    const int m = a.rows;
    const int k = a.cols;
    const int n = b.cols;

    const size_t size_a = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(float);
    const size_t size_b = static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(float);
    const size_t size_c = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    const auto start = std::chrono::high_resolution_clock::now();

    if (!check_cuda(cudaMalloc(&d_a, size_a), error, "cudaMalloc(d_a)"))
    {
        return false;
    }

    if (!check_cuda(cudaMalloc(&d_b, size_b), error, "cudaMalloc(d_b)"))
    {
        cudaFree(d_a);
        return false;
    }

    if (!check_cuda(cudaMalloc(&d_c, size_c), error, "cudaMalloc(d_c)"))
    {
        cudaFree(d_a);
        cudaFree(d_b);
        return false;
    }

    if (!check_cuda(cudaMemcpy(d_a, a.data.data(), size_a, cudaMemcpyHostToDevice), error, "cudaMemcpy H2D A"))
    {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return false;
    }

    if (!check_cuda(cudaMemcpy(d_b, b.data.data(), size_b, cudaMemcpyHostToDevice), error, "cudaMemcpy H2D B"))
    {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return false;
    }

    bool launch_ok = false;
    if (block_size == 8)
    {
        launch_ok = launch_kernel<8>(d_a, d_b, d_c, m, n, k, error);
    }
    else if (block_size == 16)
    {
        launch_ok = launch_kernel<16>(d_a, d_b, d_c, m, n, k, error);
    }
    else if (block_size == 32)
    {
        launch_ok = launch_kernel<32>(d_a, d_b, d_c, m, n, k, error);
    }
    else
    {
        error = "Unsupported block size. Allowed values: 8, 16, 32.";
    }

    if (!launch_ok)
    {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return false;
    }

    if (!check_cuda(cudaMemcpy(c.data.data(), d_c, size_c, cudaMemcpyDeviceToHost), error, "cudaMemcpy D2H C"))
    {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return false;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    const auto finish = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<float, std::milli>(finish - start).count();

    return true;
}
