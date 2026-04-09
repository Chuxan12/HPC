#include "vector_sum.hpp"

float vector_sum_cpu(const std::vector<float> &values)
{
    float sum = 0.0f;
    for (const float v : values)
    {
        sum += v;
    }
    return sum;
}
