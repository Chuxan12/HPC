#include "matmul.hpp"

#include <stdexcept>

void matmul_cpu(const Matrix &a, const Matrix &b, Matrix &c)
{
    if (a.cols != b.rows)
    {
        throw std::invalid_argument("matmul_cpu: dimension mismatch");
    }

    if (c.rows != a.rows || c.cols != b.cols)
    {
        throw std::invalid_argument("matmul_cpu: output matrix has invalid shape");
    }

    std::fill(c.data.begin(), c.data.end(), 0.0f);

    for (int i = 0; i < a.rows; ++i)
    {
        for (int k = 0; k < a.cols; ++k)
        {
            const float a_ik = a(i, k);
            const size_t b_row_offset = static_cast<size_t>(k) * static_cast<size_t>(b.cols);
            const size_t c_row_offset = static_cast<size_t>(i) * static_cast<size_t>(c.cols);
            for (int j = 0; j < b.cols; ++j)
            {
                c.data[c_row_offset + static_cast<size_t>(j)] += a_ik * b.data[b_row_offset + static_cast<size_t>(j)];
            }
        }
    }
}
