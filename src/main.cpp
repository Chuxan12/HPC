#include "vector_sum.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
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
            sizes = {1000, 10000, 50000, 100000, 500000, 1000000};
        }

        return sizes;
    }

    std::vector<float> make_random_vector(int size, uint32_t seed)
    {
        std::vector<float> values(static_cast<size_t>(size), 0.0f);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (float &v : values)
        {
            v = dist(gen);
        }

        return values;
    }

    float run_cpu_sum(const std::vector<float> &values, float &sum)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        sum = vector_sum_cpu(values);
        const auto finish = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::milli>(finish - start).count();
    }

} // namespace

int main(int argc, char **argv)
{
    const std::vector<int> sizes = parse_sizes(argc, argv);
    const std::vector<int> block_sizes = {128, 256, 512};

    // CUDA warm-up to reduce one-time context initialization impact.
    {
        const std::vector<float> warmup_values = make_random_vector(4096, 1234);
        float warmup_sum = 0.0f;
        float warmup_ms = 0.0f;
        std::string warmup_error;
        for (const int block_size : block_sizes)
        {
            vector_sum_gpu(warmup_values, block_size, warmup_sum, warmup_ms, warmup_error);
        }
    }

    std::ofstream csv_detail("results_blocksize.csv", std::ios::trunc);
    std::ofstream csv_summary("results.csv", std::ios::trunc);
    if (!csv_detail || !csv_summary)
    {
        std::cerr << "Failed to open output CSV files for writing\n";
        return 1;
    }

    csv_detail << "size,block_size,cpu_ms,gpu_ms,speedup,cpu_sum,gpu_sum,abs_diff,correct,gpu_status\n";
    csv_summary << "size,best_block_size,cpu_ms,best_gpu_ms,best_speedup,cpu_sum,best_gpu_sum,abs_diff,correct,gpu_status\n";

    std::cout << std::left << std::setw(10) << "N"
              << std::setw(10) << "Block"
              << std::setw(14) << "CPU(ms)"
              << std::setw(14) << "GPU(ms)"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "AbsDiff"
              << "Status\n";

    for (const int n : sizes)
    {
        const std::vector<float> values = make_random_vector(n, static_cast<uint32_t>(n * 17 + 1));

        float cpu_sum = 0.0f;
        const float cpu_ms = run_cpu_sum(values, cpu_sum);

        float best_gpu_ms = std::numeric_limits<float>::infinity();
        float best_speedup = 0.0f;
        float best_gpu_sum = std::numeric_limits<float>::quiet_NaN();
        float best_diff = std::numeric_limits<float>::quiet_NaN();
        int best_block = -1;
        bool best_correct = false;
        std::string best_status = "GPU_UNAVAILABLE";

        for (const int block_size : block_sizes)
        {
            float gpu_sum = 0.0f;
            float gpu_ms = 0.0f;
            std::string gpu_error;
            const bool gpu_ok = vector_sum_gpu(values, block_size, gpu_sum, gpu_ms, gpu_error);

            float diff = std::numeric_limits<float>::quiet_NaN();
            bool correct = false;
            float speedup = 0.0f;

            if (gpu_ok)
            {
                diff = std::fabs(cpu_sum - gpu_sum);
                const float tolerance = 1e-2f * static_cast<float>(n);
                correct = diff <= tolerance;
                if (gpu_ms > 0.0f)
                {
                    speedup = cpu_ms / gpu_ms;
                }

                if (correct && gpu_ms < best_gpu_ms)
                {
                    best_gpu_ms = gpu_ms;
                    best_speedup = speedup;
                    best_gpu_sum = gpu_sum;
                    best_diff = diff;
                    best_block = block_size;
                    best_correct = true;
                    best_status = "OK";
                }

                if (!best_correct && gpu_ms < best_gpu_ms)
                {
                    best_gpu_ms = gpu_ms;
                    best_speedup = speedup;
                    best_gpu_sum = gpu_sum;
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
            std::cout << std::left << std::setw(10) << n
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
                       << cpu_sum << ','
                       << (gpu_ok ? gpu_sum : -1.0f) << ','
                       << (gpu_ok ? diff : -1.0f) << ','
                       << (correct ? 1 : 0) << ','
                       << '"' << (gpu_ok ? status : gpu_error) << '"' << '\n';
        }

        csv_summary << n << ',' << best_block << ',' << cpu_ms << ','
                    << (best_block >= 0 ? best_gpu_ms : -1.0f) << ','
                    << (best_block >= 0 ? best_speedup : 0.0f) << ','
                    << cpu_sum << ','
                    << (best_block >= 0 ? best_gpu_sum : -1.0f) << ','
                    << (best_block >= 0 ? best_diff : -1.0f) << ','
                    << (best_correct ? 1 : 0) << ',' << '"' << best_status << '"' << '\n';
    }

    std::cout << "\nResults written to results.csv and results_blocksize.csv\n";
    return 0;
}
