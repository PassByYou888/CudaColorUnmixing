#pragma once
#include <vector>
#include <cstddef>
#include "types.hpp"

namespace unmixing {
namespace cuda {

struct BatchSolveInput {
    int num_pixels = 0;
    int num_layers = 0;

    std::vector<double> kernel_mu;
    std::vector<double> kernel_sigma_inv;

    std::vector<double> target_colors;
    std::vector<double> target_alphas;
    std::vector<double> initial_colors;

    std::vector<int> use_sparsity;
    std::vector<int> use_target_alphas;

    int    max_alm_iter   = params::kMaxALMIter;
    int    max_inner_iter = params::kMaxInnerEval;
    double epsilon        = params::kALMEpsilon;
    double sigma_sparsity = params::kSparsitySigma;
    double lambda_init    = params::kLambdaInit;
    double lo_init        = params::kLoInit;
};

struct BatchSolveOutput {
    std::vector<double> solution;
};

BatchSolveOutput solve_batch(const BatchSolveInput& input);

}
}