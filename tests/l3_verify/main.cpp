#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include "solver.hpp"
#include "backend.hpp"
#include "backend_cuda.hpp"

using namespace unmixing;

static double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static std::vector<double> solve_cpu_single(const std::vector<ColorKernel>& kernels,
                                            const Eigen::Vector3d& target_color) {
    SolverParams sp;
    sp.target_color = target_color;
    sp.use_sparsity = false;
    sp.use_target_alphas = false;
    const Eigen::VectorXd sol = solve_per_pixel(kernels, sp);
    std::vector<double> out(sol.size());
    for (int i = 0; i < sol.size(); ++i) out[i] = sol(i);
    return out;
}

static double energy_of(const std::vector<double>& x, int L,
                        const std::vector<ColorKernel>& kernels) {
    std::vector<double> alphas(L), colors(3 * L);
    for (int i = 0; i < L; ++i) alphas[i] = x[i];
    for (int i = 0; i < 3 * L; ++i) colors[i] = x[L + i];
    return energy_function(
        Eigen::Map<const Eigen::VectorXd>(alphas.data(), L),
        Eigen::Map<const Eigen::VectorXd>(colors.data(), 3 * L),
        kernels, 0.0, false);
}

static double constraint_norm_of(const std::vector<double>& x, int L,
                                 const Eigen::Vector3d& target) {
    Eigen::Vector3d sc = Eigen::Vector3d::Zero();
    double sa = 0.0;
    for (int i = 0; i < L; ++i) {
        sa += x[i];
        sc += x[i] * Eigen::Vector3d(x[L + i * 3 + 0], x[L + i * 3 + 1], x[L + i * 3 + 2]);
    }
    const Eigen::Vector3d gc = sc - target;
    const double c0 = gc(0) * gc(0);
    const double c1 = gc(1) * gc(1);
    const double c2 = gc(2) * gc(2);
    const double c3 = (sa - 1.0) * (sa - 1.0);
    return std::sqrt(c0 + c1 + c2 + c3);
}

int main() {
    std::mt19937 rng(98765);
    std::uniform_real_distribution<double> color_dist(0.0, 1.0);

    const int L = 4;
    const int num_pixels = 1000;
    const int dim = 4 * L;

    std::vector<ColorKernel> kernels(L);
    for (int i = 0; i < L; ++i) {
        kernels[i].mu = Eigen::Vector3d(color_dist(rng), color_dist(rng), color_dist(rng));
        Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * 0.01;
        cov(0, 1) = cov(1, 0) = 0.002;
        kernels[i].sigma_inv = cov.inverse();
    }

    std::vector<Eigen::Vector3d> colors(num_pixels);
    for (int p = 0; p < num_pixels; ++p) {
        colors[p] = Eigen::Vector3d(color_dist(rng), color_dist(rng), color_dist(rng));
    }

    std::vector<std::vector<double>> cpu_results(num_pixels);
    for (int p = 0; p < num_pixels; ++p) {
        cpu_results[p] = solve_cpu_single(kernels, colors[p]);
    }

    const IBackend& cuda = cuda_backend();

    BatchSolveRequest req;
    req.num_pixels = num_pixels;
    req.num_layers = L;
    req.kernels = kernels;
    req.target_colors.resize(num_pixels * 3);
    req.use_sparsity.assign(num_pixels, 0);
    req.use_target_alphas.assign(num_pixels, 0);
    req.max_alm_iter = params::kMaxALMIter;
    req.max_inner_iter = params::kMaxInnerEval;
    req.epsilon = params::kALMEpsilon;

    for (int p = 0; p < num_pixels; ++p) {
        req.target_colors[p * 3 + 0] = colors[p](0);
        req.target_colors[p * 3 + 1] = colors[p](1);
        req.target_colors[p * 3 + 2] = colors[p](2);
    }

    const BatchSolveResult cuda_res = cuda.solve_batch(req);
    if (cuda_res.solution.size() != (size_t)num_pixels * dim) {
        std::fprintf(stderr, "L3 FAIL: solution size mismatch\n");
        return 1;
    }

    double max_diff = 0.0;
    int worst_pixel = -1;
    std::vector<double> worst_cpu(dim), worst_cuda(dim);
    int cnt_1e_3 = 0, cnt_1e_2 = 0, cnt_1e_1 = 0;

    for (int p = 0; p < num_pixels; ++p) {
        std::vector<double> cuda_vec(dim);
        for (int i = 0; i < dim; ++i) {
            cuda_vec[i] = cuda_res.solution[(size_t)p * dim + i];
        }
        const double d = max_abs_diff(cpu_results[p], cuda_vec);
        if (d > 1e-3) ++cnt_1e_3;
        if (d > 1e-2) ++cnt_1e_2;
        if (d > 1e-1) ++cnt_1e_1;
        if (d > max_diff) {
            max_diff = d;
            worst_pixel = p;
            worst_cpu = cpu_results[p];
            worst_cuda = cuda_vec;
        }
    }

    std::printf("L3 max |x_diff|_inf = %.3e (worst pixel %d)\n", max_diff, worst_pixel);
    std::printf("   pixels > 1e-3: %d / %d\n", cnt_1e_3, num_pixels);
    std::printf("   pixels > 1e-2: %d / %d\n", cnt_1e_2, num_pixels);
    std::printf("   pixels > 1e-1: %d / %d\n", cnt_1e_1, num_pixels);

    if (worst_pixel >= 0) {
        std::printf("\nWorst pixel %d:\n", worst_pixel);
        std::printf("  cpu  x: ");
        for (int i = 0; i < dim; ++i) std::printf("%.6f ", worst_cpu[i]);
        std::printf("\n");
        std::printf("  cuda x: ");
        for (int i = 0; i < dim; ++i) std::printf("%.6f ", worst_cuda[i]);
        std::printf("\n");
        std::printf("  cpu  energy:     %.6e\n", energy_of(worst_cpu, L, kernels));
        std::printf("  cuda energy:     %.6e\n", energy_of(worst_cuda, L, kernels));
        std::printf("  cpu  constraint: %.6e\n", constraint_norm_of(worst_cpu, L, colors[worst_pixel]));
        std::printf("  cuda constraint: %.6e\n", constraint_norm_of(worst_cuda, L, colors[worst_pixel]));
        std::printf("  alpha diff: ");
        for (int i = 0; i < L; ++i) std::printf("%.3e ", std::fabs(worst_cpu[i] - worst_cuda[i]));
        std::printf("\n");
        std::printf("  color diff: ");
        for (int i = 0; i < 3 * L; ++i) std::printf("%.3e ", std::fabs(worst_cpu[L + i] - worst_cuda[L + i]));
        std::printf("\n");
    }

    std::vector<int> order(num_pixels);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    BatchSolveRequest req2 = req;
    for (int p = 0; p < num_pixels; ++p) {
        const int src = order[p];
        req2.target_colors[p * 3 + 0] = colors[src](0);
        req2.target_colors[p * 3 + 1] = colors[src](1);
        req2.target_colors[p * 3 + 2] = colors[src](2);
    }
    const BatchSolveResult cuda_res2 = cuda.solve_batch(req2);

    double order_diff = 0.0;
    for (int p = 0; p < num_pixels; ++p) {
        const int src = order[p];
        std::vector<double> a(dim), b(dim);
        for (int i = 0; i < dim; ++i) {
            a[i] = cuda_res.solution[(size_t)src * dim + i];
            b[i] = cuda_res2.solution[(size_t)p * dim + i];
        }
        order_diff = std::max(order_diff, max_abs_diff(a, b));
    }
    std::printf("L3 order-independent diff = %.3e\n", order_diff);

    if (max_diff < 1e-6 && order_diff < 1e-6) {
        std::printf("L3 PASS\n");
        return 0;
    }
    std::printf("L3 FAIL\n");
    return 1;
}