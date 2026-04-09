#include "vector_sum.hpp"

bool vector_sum_gpu(const std::vector<float> &,
                    int,
                    float &sum,
                    float &elapsed_ms,
                    std::string &error)
{
    sum = 0.0f;
    elapsed_ms = 0.0f;
    error = "CUDA is not available in this build (CMAKE_CUDA_COMPILER not found).";
    return false;
}
