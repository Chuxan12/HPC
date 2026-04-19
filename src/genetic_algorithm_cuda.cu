#include "genetic_algorithm.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <cmath>
#include <sstream>
#include <vector>

namespace
{

    __device__ double clamp_value(double x, double lo, double hi)
    {
        return x < lo ? lo : (x > hi ? hi : x);
    }

    __device__ double eval_poly(const double *coeffs, int coeff_count, double x)
    {
        double value = 0.0;
        double xp = 1.0;
        for (int i = 0; i < coeff_count; ++i)
        {
            value += coeffs[i] * xp;
            xp *= x;
        }
        return value;
    }

    __global__ void setup_rng(curandStatePhilox4_32_10_t *states, uint64_t seed, int n)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            curand_init(seed, static_cast<unsigned long long>(idx), 0ULL, &states[idx]);
        }
    }

    __global__ void evaluate_fitness(const double *population,
                                     double *fitness,
                                     const double *xs,
                                     const double *ys,
                                     int population_size,
                                     int coeff_count,
                                     int points_count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= population_size)
        {
            return;
        }

        const double *coeffs = population + idx * coeff_count;
        double max_err = 0.0;

        for (int i = 0; i < points_count; ++i)
        {
            const double err = fabs(eval_poly(coeffs, coeff_count, xs[i]) - ys[i]);
            if (err > max_err)
            {
                max_err = err;
            }
        }

        fitness[idx] = -max_err;
    }

    __device__ int tournament_select(const double *fitness,
                                     int population_size,
                                     int tournament_size,
                                     curandStatePhilox4_32_10_t &state)
    {
        int best_idx = curand(&state) % population_size;
        double best_fit = fitness[best_idx];
        for (int i = 1; i < tournament_size; ++i)
        {
            const int cand = curand(&state) % population_size;
            const double cand_fit = fitness[cand];
            if (cand_fit > best_fit)
            {
                best_fit = cand_fit;
                best_idx = cand;
            }
        }
        return best_idx;
    }

    __global__ void evolve_population(const double *population,
                                      double *next_population,
                                      const double *fitness,
                                      curandStatePhilox4_32_10_t *states,
                                      int population_size,
                                      int coeff_count,
                                      int tournament_size,
                                      double crossover_rate,
                                      double mutation_mean,
                                      double mutation_sigma,
                                      double coeff_min,
                                      double coeff_max)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= population_size || idx == 0)
        {
            return;
        }

        curandStatePhilox4_32_10_t state = states[idx];

        const int p1 = tournament_select(fitness, population_size, tournament_size, state);
        const int p2 = tournament_select(fitness, population_size, tournament_size, state);

        int cut = coeff_count;
        if (coeff_count > 1 && curand_uniform_double(&state) < crossover_rate)
        {
            cut = 1 + static_cast<int>(curand(&state) % static_cast<unsigned int>(coeff_count - 1));
        }

        const int out_base = idx * coeff_count;
        const int p1_base = p1 * coeff_count;
        const int p2_base = p2 * coeff_count;

        for (int j = 0; j < coeff_count; ++j)
        {
            next_population[out_base + j] = (j < cut) ? population[p1_base + j] : population[p2_base + j];
        }

        int mut_k = static_cast<int>(llrint(curand_normal_double(&state) * mutation_sigma + mutation_mean));
        if (mut_k < 0)
        {
            mut_k = 0;
        }
        if (mut_k > coeff_count)
        {
            mut_k = coeff_count;
        }

        for (int m = 0; m < mut_k; ++m)
        {
            const int g = curand(&state) % coeff_count;
            const double noise = curand_normal_double(&state) * 0.2;
            double gene = next_population[out_base + g] + noise;
            next_population[out_base + g] = clamp_value(gene, coeff_min, coeff_max);
        }

        states[idx] = state;
    }

    bool check_cuda(cudaError_t code, std::string &error, const char *where)
    {
        if (code == cudaSuccess)
        {
            return true;
        }
        std::ostringstream oss;
        oss << where << " failed: " << cudaGetErrorName(code) << " (" << cudaGetErrorString(code) << ")";
        error = oss.str();
        return false;
    }

} // namespace

bool run_genetic_algorithm_gpu(const GAConfig &config,
                               const std::vector<double> &xs,
                               const std::vector<double> &ys,
                               const std::vector<double> &initial_population,
                               uint32_t seed,
                               float &elapsed_ms,
                               GAResult &result,
                               std::string &error)
{
    elapsed_ms = 0.0f;
    result = GAResult{};
    error.clear();

    if (static_cast<int>(xs.size()) != config.points_count || static_cast<int>(ys.size()) != config.points_count)
    {
        error = "Dataset size does not match config.points_count.";
        return false;
    }

    const int population_size = config.population_size;
    const int coeff_count = config.coeff_count;
    const int points_count = config.points_count;

    const size_t pop_bytes = static_cast<size_t>(population_size) * static_cast<size_t>(coeff_count) * sizeof(double);
    if (initial_population.size() != static_cast<size_t>(population_size) * static_cast<size_t>(coeff_count))
    {
        error = "Initial population size does not match config.";
        return false;
    }

    const size_t fitness_bytes = static_cast<size_t>(population_size) * sizeof(double);
    const size_t points_bytes = static_cast<size_t>(points_count) * sizeof(double);
    const size_t states_bytes = static_cast<size_t>(population_size) * sizeof(curandStatePhilox4_32_10_t);

    double *d_population = nullptr;
    double *d_next_population = nullptr;
    double *d_fitness = nullptr;
    double *d_xs = nullptr;
    double *d_ys = nullptr;
    curandStatePhilox4_32_10_t *d_states = nullptr;

    cudaEvent_t start_evt = nullptr;
    cudaEvent_t stop_evt = nullptr;

    const int block = 256;
    const int grid = (population_size + block - 1) / block;

    auto free_all = [&]()
    {
        if (start_evt != nullptr)
            cudaEventDestroy(start_evt);
        if (stop_evt != nullptr)
            cudaEventDestroy(stop_evt);
        if (d_population != nullptr)
            cudaFree(d_population);
        if (d_next_population != nullptr)
            cudaFree(d_next_population);
        if (d_fitness != nullptr)
            cudaFree(d_fitness);
        if (d_xs != nullptr)
            cudaFree(d_xs);
        if (d_ys != nullptr)
            cudaFree(d_ys);
        if (d_states != nullptr)
            cudaFree(d_states);
    };

    if (!check_cuda(cudaMalloc(&d_population, pop_bytes), error, "cudaMalloc(d_population)") ||
        !check_cuda(cudaMalloc(&d_next_population, pop_bytes), error, "cudaMalloc(d_next_population)") ||
        !check_cuda(cudaMalloc(&d_fitness, fitness_bytes), error, "cudaMalloc(d_fitness)") ||
        !check_cuda(cudaMalloc(&d_xs, points_bytes), error, "cudaMalloc(d_xs)") ||
        !check_cuda(cudaMalloc(&d_ys, points_bytes), error, "cudaMalloc(d_ys)") ||
        !check_cuda(cudaMalloc(&d_states, states_bytes), error, "cudaMalloc(d_states)") ||
        !check_cuda(cudaMemcpy(d_population, initial_population.data(), pop_bytes, cudaMemcpyHostToDevice), error, "cudaMemcpy(d_population init)") ||
        !check_cuda(cudaMemcpy(d_xs, xs.data(), points_bytes, cudaMemcpyHostToDevice), error, "cudaMemcpy(d_xs)") ||
        !check_cuda(cudaMemcpy(d_ys, ys.data(), points_bytes, cudaMemcpyHostToDevice), error, "cudaMemcpy(d_ys)") ||
        !check_cuda(cudaEventCreate(&start_evt), error, "cudaEventCreate(start)") ||
        !check_cuda(cudaEventCreate(&stop_evt), error, "cudaEventCreate(stop)"))
    {
        free_all();
        return false;
    }

    if (!check_cuda(cudaEventRecord(start_evt), error, "cudaEventRecord(start)"))
    {
        free_all();
        return false;
    }

    setup_rng<<<grid, block>>>(d_states, static_cast<uint64_t>(seed), population_size);
    if (!check_cuda(cudaGetLastError(), error, "setup_rng launch"))
    {
        free_all();
        return false;
    }

    double prev_best = -1e300;
    int const_iter = 0;
    int last_generation = 0;

    for (int gen = 1; gen <= config.max_iter; ++gen)
    {
        evaluate_fitness<<<grid, block>>>(d_population, d_fitness, d_xs, d_ys, population_size, coeff_count, points_count);
        if (!check_cuda(cudaGetLastError(), error, "evaluate_fitness launch"))
        {
            free_all();
            return false;
        }

        thrust::device_ptr<double> fit_begin(d_fitness);
        thrust::device_ptr<double> best_it = thrust::max_element(fit_begin, fit_begin + population_size);
        const int best_idx = static_cast<int>(best_it - fit_begin);

        double best_fit_host = 0.0;
        if (!check_cuda(cudaMemcpy(&best_fit_host, d_fitness + best_idx, sizeof(double), cudaMemcpyDeviceToHost), error, "cudaMemcpy(best_fit)") ||
            !check_cuda(cudaMemcpy(d_next_population,
                                   d_population + static_cast<size_t>(best_idx) * static_cast<size_t>(coeff_count),
                                   static_cast<size_t>(coeff_count) * sizeof(double),
                                   cudaMemcpyDeviceToDevice),
                        error,
                        "cudaMemcpy(elitism)"))
        {
            free_all();
            return false;
        }

        if (std::fabs(best_fit_host - prev_best) < 1e-12)
        {
            ++const_iter;
        }
        else
        {
            const_iter = 0;
            prev_best = best_fit_host;
        }

        last_generation = gen;
        if (const_iter >= config.max_const_iter)
        {
            break;
        }

        evolve_population<<<grid, block>>>(d_population,
                                           d_next_population,
                                           d_fitness,
                                           d_states,
                                           population_size,
                                           coeff_count,
                                           config.tournament_size,
                                           config.crossover_rate,
                                           config.mutation_mean,
                                           std::sqrt(std::max(config.mutation_variance, 0.0)),
                                           config.coeff_min,
                                           config.coeff_max);
        if (!check_cuda(cudaGetLastError(), error, "evolve_population launch"))
        {
            free_all();
            return false;
        }

        std::swap(d_population, d_next_population);
    }

    evaluate_fitness<<<grid, block>>>(d_population, d_fitness, d_xs, d_ys, population_size, coeff_count, points_count);
    if (!check_cuda(cudaGetLastError(), error, "evaluate_fitness final launch"))
    {
        free_all();
        return false;
    }

    thrust::device_ptr<double> fit_begin(d_fitness);
    thrust::device_ptr<double> best_it = thrust::max_element(fit_begin, fit_begin + population_size);
    const int best_idx = static_cast<int>(best_it - fit_begin);

    result.best_coefficients.assign(static_cast<size_t>(coeff_count), 0.0);
    if (!check_cuda(cudaMemcpy(result.best_coefficients.data(),
                               d_population + static_cast<size_t>(best_idx) * static_cast<size_t>(coeff_count),
                               static_cast<size_t>(coeff_count) * sizeof(double),
                               cudaMemcpyDeviceToHost),
                    error,
                    "cudaMemcpy(best_coefficients)") ||
        !check_cuda(cudaMemcpy(&result.best_fitness,
                               d_fitness + best_idx,
                               sizeof(double),
                               cudaMemcpyDeviceToHost),
                    error,
                    "cudaMemcpy(best_fitness final)"))
    {
        free_all();
        return false;
    }

    result.last_generation = last_generation;

    if (!check_cuda(cudaEventRecord(stop_evt), error, "cudaEventRecord(stop)") ||
        !check_cuda(cudaEventSynchronize(stop_evt), error, "cudaEventSynchronize(stop)") ||
        !check_cuda(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt), error, "cudaEventElapsedTime"))
    {
        free_all();
        return false;
    }

    free_all();
    return true;
}
