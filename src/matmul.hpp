#pragma once

#include <string>
#include <vector>

struct Matrix
{
    int rows = 0;
    int cols = 0;
    std::vector<float> data;

    Matrix() = default;

    Matrix(int r, int c)
        : rows(r), cols(c), data(static_cast<size_t>(r) * static_cast<size_t>(c), 0.0f) {}

    float &operator()(int r, int c)
    {
        return data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)];
    }

    const float &operator()(int r, int c) const
    {
        return data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)];
    }
};

void matmul_cpu(const Matrix &a, const Matrix &b, Matrix &c);

bool matmul_gpu(const Matrix &a,
                const Matrix &b,
                Matrix &c,
                int block_size,
                float &elapsed_ms,
                std::string &error);
