#include "vector_sum.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <sstream>
#include <vector>

namespace
{

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

    template <int kBlockSize>
    __global__ void reduce_sum_kernel(const float *input, float *block_sums, int n)
    {
        __shared__ float sdata[kBlockSize];

        const unsigned int tid = threadIdx.x;
        const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

        sdata[tid] = (index < static_cast<unsigned int>(n)) ? input[index] : 0.0f;
        __syncthreads();

        for (unsigned int stride = kBlockSize / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            block_sums[blockIdx.x] = sdata[0];
        }
    }

    template <int kBlockSize>
    bool launch_kernel(const float *d_input,
                       float *d_block_sums,
                       int n,
                       int blocks,
                       std::string &error)
    {
        const dim3 block(kBlockSize);
        const dim3 grid(blocks);

        reduce_sum_kernel<kBlockSize><<<grid, block>>>(d_input, d_block_sums, n);

        if (!check_cuda(cudaGetLastError(), error, "kernel launch"))
        {
            return false;
        }

        return check_cuda(cudaDeviceSynchronize(), error, "cudaDeviceSynchronize");
    }

} // namespace

bool vector_sum_gpu(const std::vector<float> &values,
                    int block_size,
                    float &sum,
                    float &elapsed_ms,
                    std::string &error)
{
    error.clear();
    elapsed_ms = 0.0f;
    sum = 0.0f;

    if (values.empty())
    {
        return true;
    }

    if (block_size != 128 && block_size != 256 && block_size != 512)
    {
        error = "Unsupported block size. Allowed values: 128, 256, 512.";
        return false;
    }

    const int n = static_cast<int>(values.size());
    const int blocks = (n + block_size - 1) / block_size;

    const size_t input_bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t block_bytes = static_cast<size_t>(blocks) * sizeof(float);

    float *d_input = nullptr;
    float *d_block_sums = nullptr;

    const auto start = std::chrono::high_resolution_clock::now();

    if (!check_cuda(cudaMalloc(&d_input, input_bytes), error, "cudaMalloc(d_input)"))
    {
        return false;
    }

    if (!check_cuda(cudaMalloc(&d_block_sums, block_bytes), error, "cudaMalloc(d_block_sums)"))
    {
        cudaFree(d_input);
        return false;
    }

    if (!check_cuda(cudaMemcpy(d_input, values.data(), input_bytes, cudaMemcpyHostToDevice), error, "cudaMemcpy H2D input"))
    {
        cudaFree(d_input);
        cudaFree(d_block_sums);
        return false;
    }

    bool launch_ok = false;
    if (block_size == 128)
    {
        launch_ok = launch_kernel<128>(d_input, d_block_sums, n, blocks, error);
    }
    else if (block_size == 256)
    {
        launch_ok = launch_kernel<256>(d_input, d_block_sums, n, blocks, error);
    }
    else if (block_size == 512)
    {
        launch_ok = launch_kernel<512>(d_input, d_block_sums, n, blocks, error);
    }

    if (!launch_ok)
    {
        cudaFree(d_input);
        cudaFree(d_block_sums);
        return false;
    }

    std::vector<float> partial_sums(static_cast<size_t>(blocks), 0.0f);
    if (!check_cuda(cudaMemcpy(partial_sums.data(),
                               d_block_sums,
                               block_bytes,
                               cudaMemcpyDeviceToHost),
                    error,
                    "cudaMemcpy D2H block sums"))
    {
        cudaFree(d_input);
        cudaFree(d_block_sums);
        return false;
    }

    cudaFree(d_input);
    cudaFree(d_block_sums);

    for (const float v : partial_sums)
    {
        sum += v;
    }

    const auto finish = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<float, std::milli>(finish - start).count();
    return true;
}
