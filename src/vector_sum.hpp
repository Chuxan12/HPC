#pragma once

#include <string>
#include <vector>

float vector_sum_cpu(const std::vector<float> &values);

bool vector_sum_gpu(const std::vector<float> &values,
                    int block_size,
                    float &sum,
                    float &elapsed_ms,
                    std::string &error);
