#include "backend_cpu.hpp"
#include "host_ops.hpp"
#include "solver.hpp"
#include <cstdio>
#include <vector>

namespace unmixing {

namespace {

class CpuBackend final : public IBackend {
public:
    GrayImage luminance(const ColorImage& img) const override {
        return host::luminance(img);
    }

    GrayImage gradient_magnitude(const GrayImage& img) const override {
        return host::gradient_magnitude(img);
    }

    GrayImage guided_filter_kernel(const GrayImage& img,
                                   int cx, int cy, int radius,
                                   double epsilon,
                                   bool force_positive) const override {
        return host::guided_filter_kernel(img, cx, cy, radius, epsilon, force_positive);
    }

    void compute_normal_distribution(const ColorImage& img,
                                     const GrayImage& weight,
                                     Eigen::Vector3d& mu,
                                     Eigen::Matrix3d& sigma) const override {
        host::compute_normal_distribution(img, weight, mu, sigma);
    }

    GrayImage guided_filter(const GrayImage& input,
                            const GrayImage& guide_r,
                            const GrayImage& guide_g,
                            const GrayImage& guide_b,
                            int radius, double epsilon) const override {
        return host::guided_filter(input, guide_r, guide_g, guide_b, radius, epsilon);
    }

    Eigen::VectorXd solve_per_pixel(const std::vector<ColorKernel>& kernels,
                                    const SolverParams& params) const override {
        return unmixing::solve_per_pixel(kernels, params);
    }

    BatchSolveResult solve_batch(const BatchSolveRequest& req) const override {
        BatchSolveResult result;
        const int P = req.num_pixels;
        const int L = req.num_layers;
        if (P <= 0 || L <= 0) return result;

        if ((int)req.kernels.size() < L) {
            std::fprintf(stderr, "[cpu_backend] kernels too small: %zu < %d\n",
                         req.kernels.size(), L);
            return result;
        }
        if ((int)req.target_colors.size() < P * 3) {
            std::fprintf(stderr, "[cpu_backend] target_colors too small: %zu < %d\n",
                         req.target_colors.size(), P * 3);
            return result;
        }
        if ((int)req.use_sparsity.size() < P) {
            std::fprintf(stderr, "[cpu_backend] use_sparsity too small: %zu < %d\n",
                         req.use_sparsity.size(), P);
            return result;
        }
        if ((int)req.use_target_alphas.size() < P) {
            std::fprintf(stderr, "[cpu_backend] use_target_alphas too small: %zu < %d\n",
                         req.use_target_alphas.size(), P);
            return result;
        }

        bool any_ta = false;
        for (int p = 0; p < P; ++p) {
            if (req.use_target_alphas[p]) { any_ta = true; break; }
        }
        if (any_ta && (int)req.target_alphas.size() < P * L) {
            std::fprintf(stderr, "[cpu_backend] target_alphas too small: %zu < %d\n",
                         req.target_alphas.size(), P * L);
            return result;
        }
        if (!req.initial_colors.empty() && (int)req.initial_colors.size() < P * L * 3) {
            std::fprintf(stderr, "[cpu_backend] initial_colors too small: %zu < %d\n",
                         req.initial_colors.size(), P * L * 3);
            return result;
        }

        const int dim = 4 * L;
        result.solution.resize(static_cast<std::size_t>(P) * dim);

        std::vector<ColorKernel> kernels_local;
        const std::vector<ColorKernel>* kernels_ptr = &req.kernels;
        if ((int)req.kernels.size() != L) {
            kernels_local.assign(req.kernels.begin(), req.kernels.begin() + L);
            kernels_ptr = &kernels_local;
        }

        const int max_alm_iter   = (req.max_alm_iter   > 0) ? req.max_alm_iter   : params::kMaxALMIter;
        const int max_inner_iter = (req.max_inner_iter > 0) ? req.max_inner_iter : params::kMaxInnerEval;
        const double epsilon     = (req.epsilon        > 0.0) ? req.epsilon      : params::kALMEpsilon;
        const double lambda_init = (req.lambda_init    > 0.0) ? req.lambda_init  : params::kLambdaInit;
        const double lo_init     = (req.lo_init        > 0.0) ? req.lo_init      : params::kLoInit;

        for (int p = 0; p < P; ++p) {
            SolverParams sp;
            sp.target_color = Eigen::Vector3d(
                req.target_colors[p * 3 + 0],
                req.target_colors[p * 3 + 1],
                req.target_colors[p * 3 + 2]);
            sp.use_sparsity      = (req.use_sparsity[p] != 0);
            sp.use_target_alphas = (req.use_target_alphas[p] != 0);
            sp.max_alm_iter      = max_alm_iter;
            sp.max_inner_iter    = max_inner_iter;
            sp.epsilon           = epsilon;
            sp.lambda_init       = lambda_init;
            sp.lo_init           = lo_init;

            if (sp.use_target_alphas) {
                sp.target_alphas.resize(L);
                for (int i = 0; i < L; ++i) {
                    sp.target_alphas(i) = req.target_alphas[p * L + i];
                }
            }
            if (!req.initial_colors.empty()) {
                sp.initial_colors.resize(3 * L);
                for (int i = 0; i < 3 * L; ++i) {
                    sp.initial_colors(i) = req.initial_colors[static_cast<std::size_t>(p) * L * 3 + i];
                }
            }

            const Eigen::VectorXd sol = unmixing::solve_per_pixel(*kernels_ptr, sp);
            if (sol.size() != dim) {
                std::fprintf(stderr, "[cpu_backend] solve_per_pixel returned size %lld at pixel %d, expected %d\n",
                             (long long)sol.size(), p, dim);
                return BatchSolveResult{};
            }
            for (int i = 0; i < dim; ++i) {
                result.solution[static_cast<std::size_t>(p) * dim + i] = sol(i);
            }
        }
        return result;
    }
};

}

const IBackend& cpu_backend() {
    static const CpuBackend instance;
    return instance;
}

}