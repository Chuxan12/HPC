#include "genetic_algorithm.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

    bool parse_int_arg(const char *text, int &value)
    {
        try
        {
            value = std::stoi(text);
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    bool parse_double_arg(const char *text, double &value)
    {
        try
        {
            value = std::stod(text);
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    std::string coeffs_to_string(const std::vector<double> &coeffs)
    {
        std::ostringstream oss;
        oss << '[';
        for (size_t i = 0; i < coeffs.size(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << std::fixed << std::setprecision(6) << coeffs[i];
        }
        oss << ']';
        return oss.str();
    }

    double coeff_diff_l2(const std::vector<double> &a, const std::vector<double> &b)
    {
        const size_t n = std::min(a.size(), b.size());
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const double d = a[i] - b[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    }

    void print_usage(const char *exe)
    {
        std::cout << "Usage:\n"
                  << "  " << exe << " <points_count> <population_size> <Em> <Dm> <maxIter> <maxConstIter>\n\n"
                  << "Input constraints (from assignment):\n"
                  << "  points_count: 500..1000\n"
                  << "  population_size: 1000..2000\n"
                  << "  Em: mean for number of mutated genes\n"
                  << "  Dm: variance for number of mutated genes\n"
                  << "  maxIter: max generations\n"
                  << "  maxConstIter: max generations with constant best fitness\n";
    }

} // namespace

int main(int argc, char **argv)
{
    GAConfig config;
    config.coeff_count = 5;
    config.tournament_size = 3;
    config.crossover_rate = 0.9;
    config.coeff_min = -3.0;
    config.coeff_max = 3.0;
    config.x_min = -3.0;
    config.x_max = 3.0;

    if (argc != 7)
    {
        print_usage(argv[0]);
        return 1;
    }

    if (!parse_int_arg(argv[1], config.points_count) ||
        !parse_int_arg(argv[2], config.population_size) ||
        !parse_double_arg(argv[3], config.mutation_mean) ||
        !parse_double_arg(argv[4], config.mutation_variance) ||
        !parse_int_arg(argv[5], config.max_iter) ||
        !parse_int_arg(argv[6], config.max_const_iter))
    {
        std::cerr << "Failed to parse input arguments.\n";
        print_usage(argv[0]);
        return 1;
    }

    if (config.points_count < 500 || config.points_count > 1000)
    {
        std::cerr << "points_count must be in [500, 1000]\n";
        return 1;
    }
    if (config.population_size < 1000 || config.population_size > 2000)
    {
        std::cerr << "population_size must be in [1000, 2000]\n";
        return 1;
    }
    if (config.mutation_variance < 0.0)
    {
        std::cerr << "Dm (variance) must be non-negative\n";
        return 1;
    }
    if (config.max_iter <= 0 || config.max_const_iter <= 0)
    {
        std::cerr << "maxIter and maxConstIter must be positive\n";
        return 1;
    }

    std::ofstream csv_detail("results_blocksize.csv", std::ios::trunc);
    std::ofstream csv_summary("results.csv", std::ios::trunc);
    if (!csv_detail || !csv_summary)
    {
        std::cerr << "Failed to open output CSV files for writing\n";
        return 1;
    }

    csv_detail << "points_count,population_size,Em,Dm,maxIter,maxConstIter,cpu_ms,gpu_ms,speedup,cpu_best_fitness,gpu_best_fitness,fitness_abs_diff,coeff_l2_diff,gpu_last_generation,status\n";
    csv_summary << "gpu_ms,gpu_best_fitness,gpu_last_generation,gpu_coefficients\n";

    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> ground_truth_coeffs;
    std::vector<double> initial_population;

    const uint32_t dataset_seed = 2026;
    const uint32_t initial_population_seed = 777;
    const uint32_t cpu_seed = 1337;
    const uint32_t gpu_seed = 1337;

    make_dataset(config, dataset_seed, xs, ys, ground_truth_coeffs);
    make_initial_population(config, initial_population_seed, initial_population);

    float cpu_ms = 0.0f;
    const GAResult cpu_result = run_genetic_algorithm_cpu(config, xs, ys, initial_population, cpu_seed, cpu_ms);

    float gpu_ms = 0.0f;
    GAResult gpu_result;
    std::string gpu_error;
    const bool gpu_ok = run_genetic_algorithm_gpu(config, xs, ys, initial_population, gpu_seed, gpu_ms, gpu_result, gpu_error);

    const double fit_diff = gpu_ok ? std::fabs(cpu_result.best_fitness - gpu_result.best_fitness) : -1.0;
    const double coeff_l2 = gpu_ok ? coeff_diff_l2(cpu_result.best_coefficients, gpu_result.best_coefficients) : -1.0;
    const float speedup = (gpu_ok && gpu_ms > 0.0f) ? (cpu_ms / gpu_ms) : 0.0f;

    std::cout << "Input data:\n";
    std::cout << "  points_count: " << config.points_count << '\n';
    std::cout << "  population_size: " << config.population_size << '\n';
    std::cout << "  Em: " << config.mutation_mean << '\n';
    std::cout << "  Dm: " << config.mutation_variance << '\n';
    std::cout << "  maxIter: " << config.max_iter << '\n';
    std::cout << "  maxConstIter: " << config.max_const_iter << "\n\n";

    if (gpu_ok)
    {
        std::cout << "Output data (GPU):\n";
        std::cout << "  GPU processing time (ms): " << std::fixed << std::setprecision(3) << gpu_ms << '\n';
        std::cout << "  Polynomial coefficients: " << coeffs_to_string(gpu_result.best_coefficients) << '\n';
        std::cout << "  Best fitness value: " << gpu_result.best_fitness << '\n';
        std::cout << "  Last generation number: " << gpu_result.last_generation << "\n\n";

        std::cout << "CPU/GPU comparison:\n";
        std::cout << "  CPU processing time (ms): " << cpu_ms << '\n';
        std::cout << "  Speedup (CPU/GPU): " << speedup << '\n';
        std::cout << "  |fitness_cpu - fitness_gpu|: " << fit_diff << '\n';
        std::cout << "  Coeff L2 diff: " << coeff_l2 << '\n';
    }
    else
    {
        std::cout << "GPU run failed: " << gpu_error << '\n';
    }

    csv_detail << config.points_count << ','
               << config.population_size << ','
               << config.mutation_mean << ','
               << config.mutation_variance << ','
               << config.max_iter << ','
               << config.max_const_iter << ','
               << cpu_ms << ','
               << (gpu_ok ? gpu_ms : -1.0f) << ','
               << speedup << ','
               << cpu_result.best_fitness << ','
               << (gpu_ok ? gpu_result.best_fitness : -1.0) << ','
               << fit_diff << ','
               << coeff_l2 << ','
               << (gpu_ok ? gpu_result.last_generation : -1) << ','
               << '"' << (gpu_ok ? "OK" : gpu_error) << '"' << '\n';

    if (gpu_ok)
    {
        csv_summary << gpu_ms << ','
                    << gpu_result.best_fitness << ','
                    << gpu_result.last_generation << ','
                    << '"' << coeffs_to_string(gpu_result.best_coefficients) << '"' << '\n';
    }

    std::cout << "\nResults written to results.csv and results_blocksize.csv\n";
    return gpu_ok ? 0 : 2;
}
