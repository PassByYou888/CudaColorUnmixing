#include <chrono>
#include "backend_cuda.hpp"
#include "cuda_backend.hpp"
#include "batch_solver.hpp"
#include "solver.hpp"
#include <atomic>
#include <cstdio>

namespace unmixing {

namespace {
std::atomic<long long> g_luminance_calls{0};
std::atomic<long long> g_gradient_calls{0};
std::atomic<long long> g_gfk_calls{0};
std::atomic<long long> g_cnd_calls{0};
std::atomic<long long> g_gf_calls{0};
std::atomic<long long> g_gf_batch_calls{0};
std::atomic<long long> g_gf_batch_layers{0};
std::atomic<long long> g_solver_calls{0};
std::atomic<long long> g_batch_calls{0};
std::atomic<long long> g_batch_pixels{0};
}

namespace {

class CudaBackend final : public IBackend {
public:
    GrayImage luminance(const ColorImage& img) const override {
        ++g_luminance_calls;
        return cuda::luminance(img);
    }

    GrayImage gradient_magnitude(const GrayImage& img) const override {
        ++g_gradient_calls;
        return cuda::gradient_magnitude(img);
    }

    GrayImage guided_filter_kernel(const GrayImage& img,
                                   int cx, int cy, int radius,
                                   double epsilon,
                                   bool force_positive) const override {
        ++g_gfk_calls;
        return cuda::guided_filter_kernel(img, cx, cy, radius, epsilon, force_positive);
    }

    void compute_normal_distribution(const ColorImage& img,
                                     const GrayImage& weight,
                                     Eigen::Vector3d& mu,
                                     Eigen::Matrix3d& sigma) const override {
        ++g_cnd_calls;
        cuda::compute_normal_distribution(img, weight, mu, sigma);
    }

    GrayImage guided_filter(const GrayImage& input,
                            const GrayImage& guide_r,
                            const GrayImage& guide_g,
                            const GrayImage& guide_b,
                            int radius, double epsilon) const override {
        ++g_gf_calls;
        return cuda::guided_filter(input, guide_r, guide_g, guide_b, radius, epsilon);
    }

    void guided_filter_batch(const std::vector<GrayImage>& inputs,
                             const ColorImage& guide,
                             int radius, double epsilon,
                             std::vector<GrayImage>& outputs) const override {
        const int L = static_cast<int>(inputs.size());
        outputs.resize(L);
        if (L == 0) return;

        ++g_gf_batch_calls;
        g_gf_batch_layers += L;

        auto t0 = std::chrono::high_resolution_clock::now();

        const int W = inputs[0].width();
        const int H = inputs[0].height();

        cuda::GfGuidanceStats stats;
        cuda::compute_gf_guidance_stats(guide, radius, W, H, stats);
        for (int i = 0; i < L; ++i) {
            outputs[i] = cuda::guided_filter_with_stats(inputs[i], guide, stats, radius, epsilon);
        }
        cuda::free_gf_guidance_stats(stats);

        auto t1 = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr, "[gf_batch] L=%d, %.2f ms\n",
                     L, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    Eigen::VectorXd solve_per_pixel(const std::vector<ColorKernel>& kernels,
                                    const SolverParams& params) const override {
        ++g_solver_calls;
        return unmixing::solve_per_pixel(kernels, params);
    }

    BatchSolveResult solve_batch(const BatchSolveRequest& req) const override {
        ++g_batch_calls;
        g_batch_pixels += req.num_pixels;

        auto t0 = std::chrono::high_resolution_clock::now();

        const int P = req.num_pixels;
        const int L = req.num_layers;
        const int dim = 4 * L;

        BatchSolveResult result;
        if (P <= 0 || L <= 0) return result;

        if ((int)req.kernels.size() < L) {
            std::fprintf(stderr, "[cuda_backend] kernels too small: %zu < %d\n",
                         req.kernels.size(), L);
            return result;
        }
        if ((int)req.target_colors.size() < P * 3) {
            std::fprintf(stderr, "[cuda_backend] target_colors too small: %zu < %d\n",
                         req.target_colors.size(), P * 3);
            return result;
        }
        if ((int)req.use_sparsity.size() < P) {
            std::fprintf(stderr, "[cuda_backend] use_sparsity too small: %zu < %d\n",
                         req.use_sparsity.size(), P);
            return result;
        }
        if ((int)req.use_target_alphas.size() < P) {
            std::fprintf(stderr, "[cuda_backend] use_target_alphas too small: %zu < %d\n",
                         req.use_target_alphas.size(), P);
            return result;
        }

        bool any_ta = false;
        for (int p = 0; p < P; ++p) {
            if (req.use_target_alphas[p]) { any_ta = true; break; }
        }
        if (any_ta && (int)req.target_alphas.size() < P * L) {
            std::fprintf(stderr, "[cuda_backend] target_alphas too small: %zu < %d\n",
                         req.target_alphas.size(), P * L);
            return result;
        }
        if (!req.initial_colors.empty() && (int)req.initial_colors.size() < P * L * 3) {
            std::fprintf(stderr, "[cuda_backend] initial_colors too small: %zu < %d\n",
                         req.initial_colors.size(), P * L * 3);
            return result;
        }

        result.solution.resize(static_cast<std::size_t>(P) * dim);

        cuda::BatchSolveInput in;
        in.num_pixels     = P;
        in.num_layers     = L;
        in.max_alm_iter   = (req.max_alm_iter   > 0) ? req.max_alm_iter   : params::kMaxALMIter;
        in.max_inner_iter = (req.max_inner_iter > 0) ? req.max_inner_iter : params::kMaxInnerEval;
        in.epsilon        = (req.epsilon        > 0.0) ? req.epsilon      : params::kALMEpsilon;
        in.sigma_sparsity = params::kSparsitySigma;
        in.lambda_init    = (req.lambda_init    > 0.0) ? req.lambda_init  : params::kLambdaInit;
        in.lo_init        = (req.lo_init        > 0.0) ? req.lo_init      : params::kLoInit;

        in.kernel_mu.resize(L * 3);
        in.kernel_sigma_inv.resize(L * 9);
        for (int i = 0; i < L; ++i) {
            for (int j = 0; j < 3; ++j) in.kernel_mu[i * 3 + j] = req.kernels[i].mu(j);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    in.kernel_sigma_inv[i * 9 + r * 3 + c] = req.kernels[i].sigma_inv(r, c);
        }

        in.target_colors = req.target_colors;
        in.use_sparsity.assign(P, 0);
        in.use_target_alphas.assign(P, 0);

        for (int p = 0; p < P; ++p) {
            in.use_sparsity[p] = req.use_sparsity[p];
            in.use_target_alphas[p] = req.use_target_alphas[p];
        }
        if (any_ta) in.target_alphas = req.target_alphas;
        if (!req.initial_colors.empty()) in.initial_colors = req.initial_colors;

        const auto batch_out = cuda::solve_batch(in);
        if (batch_out.solution.size() != static_cast<std::size_t>(P) * dim) {
            std::fprintf(stderr, "[cuda_backend] solve_batch returned invalid solution size: %zu, expected %zu\n",
                         batch_out.solution.size(), static_cast<std::size_t>(P) * dim);
            result.solution.clear();
            return result;
        }
        result.solution = batch_out.solution;

        auto t1 = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr, "[solve_batch] P=%d, %.2f ms\n",
                     P, std::chrono::duration<double, std::milli>(t1 - t0).count());

        return result;
    }
};

}

const IBackend& cuda_backend() {
    static const CudaBackend instance;
    return instance;
}

void cuda_backend_dump_stats() {
    std::printf("[cuda] backend call counts:\n");
    std::printf("[cuda]   luminance                  (cuda) : %lld\n", (long long)g_luminance_calls.load());
    std::printf("[cuda]   gradient_magnitude         (cuda) : %lld\n", (long long)g_gradient_calls.load());
    std::printf("[cuda]   guided_filter_kernel       (cuda) : %lld\n", (long long)g_gfk_calls.load());
    std::printf("[cuda]   compute_normal_distribution(cuda) : %lld\n", (long long)g_cnd_calls.load());
    std::printf("[cuda]   guided_filter              (cuda) : %lld\n", (long long)g_gf_calls.load());
    std::printf("[cuda]   guided_filter_batch        (cuda) : %lld calls, %lld layers\n",
                (long long)g_gf_batch_calls.load(), (long long)g_gf_batch_layers.load());
    std::printf("[cuda]   solve_per_pixel            (cpu ) : %lld\n", (long long)g_solver_calls.load());
    std::printf("[cuda]   solve_batch                (cuda) : %lld calls, %lld pixels\n",
                (long long)g_batch_calls.load(), (long long)g_batch_pixels.load());
}

}