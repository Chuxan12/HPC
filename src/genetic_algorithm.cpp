#include "genetic_algorithm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

namespace
{

    using Coeffs = std::vector<double>;

    double clamp_value(double x, double lo, double hi)
    {
        return std::max(lo, std::min(x, hi));
    }

    double eval_poly(const Coeffs &coeffs, double x)
    {
        double value = 0.0;
        double xp = 1.0;
        for (const double c : coeffs)
        {
            value += c * xp;
            xp *= x;
        }
        return value;
    }

    double max_abs_error_cost(const Coeffs &coeffs,
                              const std::vector<double> &xs,
                              const std::vector<double> &ys)
    {
        double max_err = 0.0;
        const size_t n = xs.size();
        for (size_t i = 0; i < n; ++i)
        {
            const double err = std::fabs(eval_poly(coeffs, xs[i]) - ys[i]);
            if (err > max_err)
            {
                max_err = err;
            }
        }
        return max_err;
    }

    double fitness_of(const Coeffs &coeffs,
                      const std::vector<double> &xs,
                      const std::vector<double> &ys)
    {
        return -max_abs_error_cost(coeffs, xs, ys);
    }

    int tournament_select(const std::vector<double> &fitness,
                          std::mt19937 &rng,
                          int tournament_size)
    {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(fitness.size()) - 1);
        int best_idx = pick(rng);
        double best_fit = fitness[best_idx];

        for (int i = 1; i < tournament_size; ++i)
        {
            const int idx = pick(rng);
            if (fitness[idx] > best_fit)
            {
                best_fit = fitness[idx];
                best_idx = idx;
            }
        }

        return best_idx;
    }

    int sample_mutated_genes(double mean, double variance, int coeff_count, std::mt19937 &rng)
    {
        const double sigma = std::sqrt(std::max(variance, 0.0));
        std::normal_distribution<double> dist(mean, sigma);
        int k = static_cast<int>(std::llround(dist(rng)));
        k = std::max(0, std::min(k, coeff_count));
        return k;
    }

} // namespace

void make_dataset(const GAConfig &config,
                  uint32_t seed,
                  std::vector<double> &xs,
                  std::vector<double> &ys,
                  std::vector<double> &ground_truth_coeffs)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> x_dist(config.x_min, config.x_max);
    std::normal_distribution<double> noise(0.0, 0.15);

    ground_truth_coeffs = {1.5, -2.1, 0.8, -0.12, 0.03};
    if (config.coeff_count < static_cast<int>(ground_truth_coeffs.size()))
    {
        ground_truth_coeffs.resize(static_cast<size_t>(config.coeff_count));
    }
    if (config.coeff_count > static_cast<int>(ground_truth_coeffs.size()))
    {
        ground_truth_coeffs.resize(static_cast<size_t>(config.coeff_count), 0.0);
    }

    xs.assign(static_cast<size_t>(config.points_count), 0.0);
    ys.assign(static_cast<size_t>(config.points_count), 0.0);

    for (int i = 0; i < config.points_count; ++i)
    {
        const double x = x_dist(rng);
        xs[static_cast<size_t>(i)] = x;
        ys[static_cast<size_t>(i)] = eval_poly(ground_truth_coeffs, x) + noise(rng);
    }
}

void make_initial_population(const GAConfig &config,
                             uint32_t seed,
                             std::vector<double> &initial_population)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> coeff_dist(config.coeff_min, config.coeff_max);

    const size_t total = static_cast<size_t>(config.population_size) * static_cast<size_t>(config.coeff_count);
    initial_population.assign(total, 0.0);

    for (size_t i = 0; i < total; ++i)
    {
        initial_population[i] = coeff_dist(rng);
    }
}

GAResult run_genetic_algorithm_cpu(const GAConfig &config,
                                   const std::vector<double> &xs,
                                   const std::vector<double> &ys,
                                   const std::vector<double> &initial_population,
                                   uint32_t seed,
                                   float &elapsed_ms)
{
    const auto start = std::chrono::high_resolution_clock::now();

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::normal_distribution<double> mutation_noise(0.0, 0.2);
    std::uniform_int_distribution<int> gene_pick(0, std::max(0, config.coeff_count - 1));

    std::vector<Coeffs> population(static_cast<size_t>(config.population_size),
                                   Coeffs(static_cast<size_t>(config.coeff_count), 0.0));

    const size_t expected = static_cast<size_t>(config.population_size) * static_cast<size_t>(config.coeff_count);
    if (initial_population.size() != expected)
    {
        elapsed_ms = 0.0f;
        return GAResult{};
    }

    for (int i = 0; i < config.population_size; ++i)
    {
        for (int j = 0; j < config.coeff_count; ++j)
        {
            population[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                initial_population[static_cast<size_t>(i) * static_cast<size_t>(config.coeff_count) + static_cast<size_t>(j)];
        }
    }

    std::vector<double> fitness(static_cast<size_t>(config.population_size), 0.0);
    for (int i = 0; i < config.population_size; ++i)
    {
        fitness[static_cast<size_t>(i)] = fitness_of(population[static_cast<size_t>(i)], xs, ys);
    }

    double prev_best = -1e300;
    int const_iter = 0;
    int last_generation = 0;

    for (int gen = 1; gen <= config.max_iter; ++gen)
    {
        const auto best_it = std::max_element(fitness.begin(), fitness.end());
        const int best_idx = static_cast<int>(std::distance(fitness.begin(), best_it));
        const double best_fit = *best_it;

        if (std::fabs(best_fit - prev_best) < 1e-12)
        {
            ++const_iter;
        }
        else
        {
            const_iter = 0;
            prev_best = best_fit;
        }

        if (const_iter >= config.max_const_iter)
        {
            last_generation = gen;
            break;
        }

        std::vector<Coeffs> next_population;
        next_population.reserve(population.size());
        next_population.push_back(population[static_cast<size_t>(best_idx)]);

        while (static_cast<int>(next_population.size()) < config.population_size)
        {
            const int p1_idx = tournament_select(fitness, rng, config.tournament_size);
            const int p2_idx = tournament_select(fitness, rng, config.tournament_size);

            Coeffs child1 = population[static_cast<size_t>(p1_idx)];
            Coeffs child2 = population[static_cast<size_t>(p2_idx)];

            if (config.coeff_count > 1 && prob(rng) < config.crossover_rate)
            {
                const int cut = 1 + (rng() % static_cast<uint32_t>(config.coeff_count - 1));
                for (int j = cut; j < config.coeff_count; ++j)
                {
                    std::swap(child1[static_cast<size_t>(j)], child2[static_cast<size_t>(j)]);
                }
            }

            const int k1 = sample_mutated_genes(config.mutation_mean, config.mutation_variance, config.coeff_count, rng);
            const int k2 = sample_mutated_genes(config.mutation_mean, config.mutation_variance, config.coeff_count, rng);

            for (int m = 0; m < k1; ++m)
            {
                const int g = gene_pick(rng);
                child1[static_cast<size_t>(g)] = clamp_value(child1[static_cast<size_t>(g)] + mutation_noise(rng),
                                                             config.coeff_min,
                                                             config.coeff_max);
            }

            for (int m = 0; m < k2; ++m)
            {
                const int g = gene_pick(rng);
                child2[static_cast<size_t>(g)] = clamp_value(child2[static_cast<size_t>(g)] + mutation_noise(rng),
                                                             config.coeff_min,
                                                             config.coeff_max);
            }

            next_population.push_back(std::move(child1));
            if (static_cast<int>(next_population.size()) < config.population_size)
            {
                next_population.push_back(std::move(child2));
            }
        }

        population = std::move(next_population);
        for (int i = 0; i < config.population_size; ++i)
        {
            fitness[static_cast<size_t>(i)] = fitness_of(population[static_cast<size_t>(i)], xs, ys);
        }

        last_generation = gen;
    }

    const auto best_it = std::max_element(fitness.begin(), fitness.end());
    const int best_idx = static_cast<int>(std::distance(fitness.begin(), best_it));

    const auto finish = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<float, std::milli>(finish - start).count();

    GAResult result;
    result.best_fitness = fitness[static_cast<size_t>(best_idx)];
    result.best_coefficients = population[static_cast<size_t>(best_idx)];
    result.last_generation = last_generation;
    return result;
}
