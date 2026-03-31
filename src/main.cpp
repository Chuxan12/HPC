#include "matmul.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{

    std::vector<int> parse_sizes(int argc, char **argv)
    {
        std::vector<int> sizes;

        for (int i = 1; i < argc; ++i)
        {
            try
            {
                const int value = std::stoi(argv[i]);
                if (value <= 0)
                {
                    throw std::invalid_argument("size must be positive");
                }
                sizes.push_back(value);
            }
            catch (const std::exception &)
            {
                std::cerr << "Invalid size argument: '" << argv[i] << "'. Use positive integers.\n";
                std::exit(1);
            }
        }

        if (sizes.empty())
        {
            sizes = {100, 300, 500, 700, 1000, 1500, 2000};
        }

        return sizes;
    }

    Matrix make_random_matrix(int rows, int cols, uint32_t seed)
    {
        Matrix m(rows, cols);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (auto &v : m.data)
        {
            v = dist(gen);
        }

        return m;
    }

    float run_cpu(const Matrix &a, const Matrix &b, Matrix &c)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        matmul_cpu(a, b, c);
        const auto finish = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::milli>(finish - start).count();
    }

    float max_abs_diff(const Matrix &x, const Matrix &y)
    {
        if (x.rows != y.rows || x.cols != y.cols)
        {
            return std::numeric_limits<float>::infinity();
        }

        float diff = 0.0f;
        for (size_t i = 0; i < x.data.size(); ++i)
        {
            diff = std::max(diff, std::fabs(x.data[i] - y.data[i]));
        }

        return diff;
    }

}

int main(int argc, char **argv)
{
    const std::vector<int> sizes = parse_sizes(argc, argv);
    const std::vector<int> block_sizes = {8, 16, 32};

    {
        Matrix wa = make_random_matrix(64, 64, 1234);
        Matrix wb = make_random_matrix(64, 64, 4321);
        Matrix wc(64, 64);
        float warmup_ms = 0.0f;
        std::string warmup_error;
        for (const int block_size : block_sizes)
        {
            matmul_gpu(wa, wb, wc, block_size, warmup_ms, warmup_error);
        }
    }

    std::ofstream csv_detail("results_blocksize.csv", std::ios::trunc);
    std::ofstream csv_summary("results.csv", std::ios::trunc);
    if (!csv_detail || !csv_summary)
    {
        std::cerr << "Failed to open output CSV files for writing\n";
        return 1;
    }

    csv_detail << "size,block_size,cpu_ms,gpu_ms,speedup,max_abs_diff,correct,gpu_status\n";
    csv_summary << "size,best_block_size,cpu_ms,best_gpu_ms,best_speedup,max_abs_diff,correct,gpu_status\n";

    std::cout << std::left << std::setw(8) << "N"
              << std::setw(10) << "Block"
              << std::setw(14) << "CPU(ms)"
              << std::setw(14) << "GPU(ms)"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "MaxDiff"
              << "Status\n";

    for (const int n : sizes)
    {
        Matrix a = make_random_matrix(n, n, static_cast<uint32_t>(n * 17 + 1));
        Matrix b = make_random_matrix(n, n, static_cast<uint32_t>(n * 17 + 2));

        Matrix c_cpu(n, n);
        const float cpu_ms = run_cpu(a, b, c_cpu);

        float best_gpu_ms = std::numeric_limits<float>::infinity();
        float best_speedup = 0.0f;
        float best_diff = std::numeric_limits<float>::quiet_NaN();
        int best_block = -1;
        bool best_correct = false;
        std::string best_status = "GPU_UNAVAILABLE";

        for (const int block_size : block_sizes)
        {
            Matrix c_gpu(n, n);
            float gpu_ms = 0.0f;
            std::string gpu_error;
            const bool gpu_ok = matmul_gpu(a, b, c_gpu, block_size, gpu_ms, gpu_error);

            float diff = std::numeric_limits<float>::quiet_NaN();
            bool correct = false;
            float speedup = 0.0f;

            if (gpu_ok)
            {
                diff = max_abs_diff(c_cpu, c_gpu);
                correct = diff < 1e-2f;
                if (gpu_ms > 0.0f)
                {
                    speedup = cpu_ms / gpu_ms;
                }

                if (correct && gpu_ms < best_gpu_ms)
                {
                    best_gpu_ms = gpu_ms;
                    best_speedup = speedup;
                    best_diff = diff;
                    best_block = block_size;
                    best_correct = true;
                    best_status = "OK";
                }

                if (!best_correct && gpu_ms < best_gpu_ms)
                {
                    best_gpu_ms = gpu_ms;
                    best_speedup = speedup;
                    best_diff = diff;
                    best_block = block_size;
                    best_status = "MISMATCH";
                }
            }
            else if (!best_correct && best_block == -1)
            {
                best_status = "GPU_UNAVAILABLE";
            }

            const std::string status = gpu_ok ? (correct ? "OK" : "MISMATCH") : "GPU_UNAVAILABLE";
            std::cout << std::left << std::setw(8) << n
                      << std::setw(10) << block_size
                      << std::setw(14) << std::fixed << std::setprecision(3) << cpu_ms
                      << std::setw(14) << (gpu_ok ? gpu_ms : -1.0f)
                      << std::setw(12) << (gpu_ok ? speedup : 0.0f)
                      << std::setw(14) << (gpu_ok ? diff : -1.0f)
                      << status;

            if (!gpu_ok)
            {
                std::cout << " (" << gpu_error << ")";
            }
            std::cout << '\n';

            csv_detail << n << ',' << block_size << ',' << cpu_ms << ','
                       << (gpu_ok ? gpu_ms : -1.0f) << ','
                       << (gpu_ok ? speedup : 0.0f) << ','
                       << (gpu_ok ? diff : -1.0f) << ','
                       << (correct ? 1 : 0) << ','
                       << '"' << (gpu_ok ? status : gpu_error) << '"' << '\n';
        }

        csv_summary << n << ',' << best_block << ',' << cpu_ms << ','
                    << (best_block >= 0 ? best_gpu_ms : -1.0f) << ','
                    << (best_block >= 0 ? best_speedup : 0.0f) << ','
                    << (best_block >= 0 ? best_diff : -1.0f) << ','
                    << (best_correct ? 1 : 0) << ',' << '"' << best_status << '"' << '\n';
    }

    std::cout << "\nResults written to results.csv and results_blocksize.csv\n";
    return 0;
}
