#include "matmul.hpp"

bool matmul_gpu(const Matrix &,
                const Matrix &,
                Matrix &,
                int,
                float &elapsed_ms,
                std::string &error)
{
    elapsed_ms = 0.0f;
    error = "CUDA is not available in this build (CMAKE_CUDA_COMPILER not found).";
    return false;
}
