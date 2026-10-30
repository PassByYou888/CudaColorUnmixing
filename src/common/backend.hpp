#pragma once
#include <vector>
#include <Eigen/Core>
#include "image.hpp"
#include "solver.hpp"
#include "types.hpp"

namespace unmixing {

struct BatchSolveRequest {
    int num_pixels = 0;
    int num_layers = 0;
    std::vector<double> target_colors;
    std::vector<double> target_alphas;
    std::vector<double> initial_colors;
    std::vector<int>    use_sparsity;
    std::vector<int>    use_target_alphas;
    std::vector<ColorKernel> kernels;
    int    max_alm_iter   = params::kMaxALMIter;
    int    max_inner_iter = params::kMaxInnerEval;
    double epsilon        = params::kALMEpsilon;
    double lambda_init    = params::kLambdaInit;
    double lo_init        = params::kLoInit;
};

struct BatchSolveResult {
    std::vector<double> solution;
};

class IBackend {
public:
    virtual ~IBackend() = default;

    virtual GrayImage luminance(const ColorImage& img) const = 0;
    virtual GrayImage gradient_magnitude(const GrayImage& img) const = 0;

    virtual GrayImage guided_filter_kernel(const GrayImage& img,
                                           int cx, int cy, int radius,
                                           double epsilon,
                                           bool force_positive) const = 0;

    virtual void compute_normal_distribution(const ColorImage& img,
                                             const GrayImage& weight,
                                             Eigen::Vector3d& mu,
                                             Eigen::Matrix3d& sigma) const = 0;

    virtual GrayImage guided_filter(const GrayImage& input,
                                    const GrayImage& guide_r,
                                    const GrayImage& guide_g,
                                    const GrayImage& guide_b,
                                    int radius, double epsilon) const = 0;

    virtual void guided_filter_batch(const std::vector<GrayImage>& inputs,
                                     const ColorImage& guide,
                                     int radius, double epsilon,
                                     std::vector<GrayImage>& outputs) const {
        outputs.resize(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            outputs[i] = guided_filter(inputs[i],
                                       guide.r(), guide.g(), guide.b(),
                                       radius, epsilon);
        }
    }

    virtual Eigen::VectorXd solve_per_pixel(const std::vector<ColorKernel>& kernels,
                                            const SolverParams& params) const = 0;

    virtual BatchSolveResult solve_batch(const BatchSolveRequest& req) const = 0;
};

}