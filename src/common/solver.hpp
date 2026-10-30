#pragma once
#include <vector>
#include <Eigen/Core>
#include "types.hpp"

namespace unmixing {

struct SolverParams {
    Eigen::Vector3d target_color      = Eigen::Vector3d::Zero();
    bool            use_sparsity      = false;
    bool            use_target_alphas = false;
    Eigen::VectorXd target_alphas;
    Eigen::VectorXd initial_colors;
    double          lambda_init    = params::kLambdaInit;
    double          lo_init        = params::kLoInit;
    double          sigma          = params::kSparsitySigma;
    int             max_alm_iter   = params::kMaxALMIter;
    int             max_inner_iter = params::kMaxInnerEval;
    double          epsilon        = params::kALMEpsilon;
};

Eigen::VectorXd solve_per_pixel(const std::vector<ColorKernel>& kernels,
                                const SolverParams& params);

double energy_function(const Eigen::VectorXd& alphas,
                       const Eigen::VectorXd& colors,
                       const std::vector<ColorKernel>& kernels,
                       double sigma,
                       bool use_sparsity);

}