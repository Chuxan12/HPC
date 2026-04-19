#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GAConfig
{
    int population_size = 1000;
    int coeff_count = 4;
    int max_iter = 300;
    int max_const_iter = 50;
    int points_count = 600;
    int tournament_size = 3;
    double crossover_rate = 0.9;
    double mutation_mean = 1.0;
    double mutation_variance = 1.0;
    double coeff_min = -3.0;
    double coeff_max = 3.0;
    double x_min = -3.0;
    double x_max = 3.0;
};

struct GAResult
{
    double best_fitness = 0.0;
    std::vector<double> best_coefficients;
    int last_generation = 0;
};

void make_dataset(const GAConfig &config,
                  uint32_t seed,
                  std::vector<double> &xs,
                  std::vector<double> &ys,
                  std::vector<double> &ground_truth_coeffs);

void make_initial_population(const GAConfig &config,
                             uint32_t seed,
                             std::vector<double> &initial_population);

GAResult run_genetic_algorithm_cpu(const GAConfig &config,
                                   const std::vector<double> &xs,
                                   const std::vector<double> &ys,
                                   const std::vector<double> &initial_population,
                                   uint32_t seed,
                                   float &elapsed_ms);

bool run_genetic_algorithm_gpu(const GAConfig &config,
                               const std::vector<double> &xs,
                               const std::vector<double> &ys,
                               const std::vector<double> &initial_population,
                               uint32_t seed,
                               float &elapsed_ms,
                               GAResult &result,
                               std::string &error);
