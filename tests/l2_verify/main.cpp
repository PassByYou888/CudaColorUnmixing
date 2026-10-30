#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "solver.hpp"
#include "backend.hpp"
#include "backend_cpu.hpp"
#include "backend_cuda.hpp"

using namespace unmixing;

namespace {

double seq_dot(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

double seq_sqnorm(const double* a, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * a[i];
    return s;
}

double seq_norm(const double* a, int n) {
    return std::sqrt(seq_sqnorm(a, n));
}

void seq_constraint(const double* x, int L, const Eigen::Vector3d& target, double* cons) {
    const double* alphas = x;
    const double* colors = x + L;
    double sc[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < L; ++i) {
        sc[0] += alphas[i] * colors[i * 3 + 0];
        sc[1] += alphas[i] * colors[i * 3 + 1];
        sc[2] += alphas[i] * colors[i * 3 + 2];
    }
    cons[0] = (sc[0] - target(0)) * (sc[0] - target(0));
    cons[1] = (sc[1] - target(1)) * (sc[1] - target(1));
    cons[2] = (sc[2] - target(2)) * (sc[2] - target(2));
    double sa = 0.0;
    for (int i = 0; i < L; ++i) sa += alphas[i];
    cons[3] = (sa - 1.0) * (sa - 1.0);
}

double seq_energy(const double* x, int L, const std::vector<ColorKernel>& kernels) {
    const double* alphas = x;
    const double* colors = x + L;
    double e = 0.0;
    for (int i = 0; i < L; ++i) {
        const Eigen::Vector3d c(colors[i * 3 + 0], colors[i * 3 + 1], colors[i * 3 + 2]);
        e += alphas[i] * kernels[i].squared_mahalanobis(c);
    }
    if (!std::isfinite(e)) e = 1e10;
    return e;
}

void seq_grad_energy(const double* x, int L, const std::vector<ColorKernel>& kernels, double* g) {
    const double* alphas = x;
    const double* colors = x + L;
    for (int i = 0; i < L; ++i) {
        const Eigen::Vector3d c(colors[i * 3 + 0], colors[i * 3 + 1], colors[i * 3 + 2]);
        const Eigen::Vector3d d = c - kernels[i].mu;
        const Eigen::Vector3d sd = kernels[i].sigma_inv * d;
        g[i] = kernels[i].squared_mahalanobis(c);
        g[L + i * 3 + 0] = 2.0 * alphas[i] * sd(0);
        g[L + i * 3 + 1] = 2.0 * alphas[i] * sd(1);
        g[L + i * 3 + 2] = 2.0 * alphas[i] * sd(2);
    }
    for (int i = 0; i < 4 * L; ++i)
        if (!std::isfinite(g[i])) g[i] = 0.0;
}

void seq_grad_constraint(const double* x, int L, const Eigen::Vector3d& target,
                         const double* cons, const double* lambda, double lo, double* g) {
    const double* alphas = x;
    const double* colors = x + L;
    double sc[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < L; ++i) {
        sc[0] += alphas[i] * colors[i * 3 + 0];
        sc[1] += alphas[i] * colors[i * 3 + 1];
        sc[2] += alphas[i] * colors[i * 3 + 2];
    }
    double sa = 0.0;
    for (int i = 0; i < L; ++i) sa += alphas[i];

    for (int i = 0; i < L; ++i) {
        const double a = alphas[i];
        const double* col = colors + i * 3;
        double dgc_da[3];
        for (int j = 0; j < 3; ++j) dgc_da[j] = 2.0 * col[j] * (sc[j] - target(j));
        const double dga_da = 2.0 * (sa - 1.0);
        g[i] = lambda[0] * dgc_da[0] + lambda[1] * dgc_da[1] + lambda[2] * dgc_da[2]
             + lambda[3] * dga_da
             + lo * (cons[0] * dgc_da[0] + cons[1] * dgc_da[1] + cons[2] * dgc_da[2] + cons[3] * dga_da);
        for (int j = 0; j < 3; ++j) {
            const double dgr = 2.0 * a * (sc[j] - target(j));
            g[L + i * 3 + j] = lambda[j] * dgr + lo * cons[j] * dgr;
        }
    }
    for (int i = 0; i < 4 * L; ++i)
        if (!std::isfinite(g[i])) g[i] = 0.0;
}

double seq_fg(const double* x, int L, const std::vector<ColorKernel>& kernels,
              const Eigen::Vector3d& target, const double* lambda, double lo, double* g) {
    double cons[4];
    seq_constraint(x, L, target, cons);
    const double e = seq_energy(x, L, kernels);
    double p = 0.0;
    for (int j = 0; j < 4; ++j) p += lambda[j] * cons[j] + 0.5 * lo * cons[j] * cons[j];
    if (g) {
        std::fill(g, g + 4 * L, 0.0);
        seq_grad_energy(x, L, kernels, g);
        std::vector<double> gc(static_cast<size_t>(4 * L), 0.0);
        seq_grad_constraint(x, L, target, cons, lambda, lo, gc.data());
        for (int i = 0; i < 4 * L; ++i) g[i] += gc[i];
    }
    return e + p;
}

std::vector<double> seq_lbfgs(const std::vector<double>& x0,
                              const std::vector<ColorKernel>& kernels,
                              const Eigen::Vector3d& target,
                              const double* lambda, double lo,
                              int max_iter) {
    const int L = static_cast<int>(kernels.size());
    const int dim = 4 * L;
    const int m = 6;

    std::vector<double> x = x0;
    std::vector<double> S(m * dim), Y(m * dim);
    std::vector<double> rho(m, 0.0);
    int k = 0;
    std::vector<double> g(dim);
    double f = seq_fg(x.data(), L, kernels, target, lambda, lo, g.data());

    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<double> pg = g;
        for (int i = 0; i < dim; ++i) {
            if ((x[i] <= 0.0 && pg[i] > 0.0) || (x[i] >= 1.0 && pg[i] < 0.0)) pg[i] = 0.0;
        }
        if (seq_norm(pg.data(), dim) < 1e-10) break;

        const int m_used = std::min(k, m);
        std::vector<double> q = pg;
        std::vector<double> alpha(m_used, 0.0);
        for (int j = m_used - 1; j >= 0; --j) {
            const int idx = (k - m_used + j) % m;
            alpha[j] = rho[idx] * seq_dot(S.data() + idx * dim, q.data(), dim);
            for (int i = 0; i < dim; ++i) q[i] -= alpha[j] * Y[idx * dim + i];
        }
        double gamma = 1.0;
        if (m_used > 0) {
            const int idx_last = (k - 1) % m;
            const double sy = seq_dot(S.data() + idx_last * dim, Y.data() + idx_last * dim, dim);
            const double yy = seq_sqnorm(Y.data() + idx_last * dim, dim);
            if (sy > 1e-10) gamma = sy / yy;
        }
        std::vector<double> d(dim);
        for (int i = 0; i < dim; ++i) d[i] = gamma * q[i];
        for (int j = 0; j < m_used; ++j) {
            const int idx = (k - m_used + j) % m;
            const double beta = rho[idx] * seq_dot(Y.data() + idx * dim, d.data(), dim);
            for (int i = 0; i < dim; ++i) d[i] += (alpha[j] - beta) * S[idx * dim + i];
        }
        for (int i = 0; i < dim; ++i) d[i] = -d[i];

        double dphi0 = seq_dot(g.data(), d.data(), dim);
        if (!std::isfinite(dphi0) || dphi0 >= 0.0) {
            k = 0;
            for (int i = 0; i < dim; ++i) d[i] = -pg[i];
        }

        double step = 1.0;
        std::vector<double> x_new(dim), g_new(dim);
        double f_new = f;
        bool accepted = false;
        for (int ls = 0; ls < 30; ++ls) {
            for (int i = 0; i < dim; ++i)
                x_new[i] = std::clamp(x[i] + step * d[i], 0.0, 1.0);
            f_new = seq_fg(x_new.data(), L, kernels, target, lambda, lo, g_new.data());
            double actual = 0.0;
            for (int i = 0; i < dim; ++i) actual += g[i] * (x_new[i] - x[i]);
            if (std::isfinite(f_new) && f_new <= f + 1e-4 * actual) { accepted = true; break; }
            step *= 0.5;
        }
        if (!accepted) { x_new = x; g_new = g; f_new = f; }

        std::vector<double> s(dim), y(dim);
        for (int i = 0; i < dim; ++i) { s[i] = x_new[i] - x[i]; y[i] = g_new[i] - g[i]; }
        const double sy = seq_dot(s.data(), y.data(), dim);
        if (sy > 1e-10 && std::isfinite(sy)) {
            const int slot = k % m;
            for (int i = 0; i < dim; ++i) { S[slot * dim + i] = s[i]; Y[slot * dim + i] = y[i]; }
            rho[slot] = 1.0 / sy;
            ++k;
        } else k = 0;

        const double dx = seq_norm(s.data(), dim);
        const double df = std::fabs(f - f_new);
        x = x_new; g = g_new; f = f_new;
        if (dx < 1e-8 || df < 1e-12) break;
    }
    return x;
}

std::vector<double> seq_solve_per_pixel(const std::vector<ColorKernel>& kernels,
                                        const Eigen::Vector3d& target) {
    const int L = static_cast<int>(kernels.size());
    const int dim = 4 * L;

    std::vector<double> x(dim, 0.0);
    {
        double md = 1e30;
        int closest = 0;
        for (int i = 0; i < L; ++i) {
            const double d = kernels[i].squared_mahalanobis(target);
            if (d < md) { md = d; closest = i; }
        }
        x[closest] = 1.0;
        for (int i = 0; i < L; ++i) {
            const Eigen::Vector3d c = (i == closest) ? target : kernels[i].mu;
            for (int j = 0; j < 3; ++j)
                x[L + i * 3 + j] = std::clamp(c(j), 0.0, 1.0);
        }
    }

    double lambda[4] = {0.1, 0.1, 0.1, 0.1};
    double lo = 0.1;
    double cons_old[4];
    seq_constraint(x.data(), L, target, cons_old);

    for (int outer = 0; outer < 100; ++outer) {
        const std::vector<double> x_prev = x;
        x = seq_lbfgs(x, kernels, target, lambda, lo, 200);

        double cons_new[4];
        seq_constraint(x.data(), L, target, cons_new);

        for (int j = 0; j < 4; ++j) lambda[j] += lo * cons_new[j];
        const double lambda_norm = seq_norm(lambda, 4);
        if (!std::isfinite(lambda_norm)) {
            for (int j = 0; j < 4; ++j) lambda[j] = 0.1;
        }
        const double cons_norm = seq_norm(cons_new, 4);
        const double old_norm = seq_norm(cons_old, 4);
        if (!std::isfinite(cons_norm) || cons_norm > 0.25 * old_norm) lo *= 10.0;
        if (!std::isfinite(lo) || lo > 1e6) lo = 1e6;

        double dx_sq = 0.0;
        for (int i = 0; i < dim; ++i) {
            const double d = x[i] - x_prev[i];
            dx_sq += d * d;
        }
        for (int j = 0; j < 4; ++j) cons_old[j] = cons_new[j];

        if (dx_sq < 1e-8 && cons_norm < 1e-8) break;
    }

    for (int i = 0; i < L; ++i) {
        double a = std::clamp(x[i], 0.0, 1.0);
        x[i] = a;
        if (a < 1e-4) {
            x[L + i * 3 + 0] = kernels[i].mu(0);
            x[L + i * 3 + 1] = kernels[i].mu(1);
            x[L + i * 3 + 2] = kernels[i].mu(2);
        } else {
            for (int j = 0; j < 3; ++j)
                x[L + i * 3 + j] = std::clamp(x[L + i * 3 + j], 0.0, 1.0);
        }
    }
    return x;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double mx = 0.0;
    for (size_t i = 0; i < a.size(); ++i) mx = std::max(mx, std::fabs(a[i] - b[i]));
    return mx;
}

double final_energy(const std::vector<double>& x, int L, const std::vector<ColorKernel>& kernels) {
    std::vector<double> alphas(L), colors(3 * L);
    for (int i = 0; i < L; ++i) alphas[i] = x[i];
    for (int i = 0; i < 3 * L; ++i) colors[i] = x[L + i];
    return energy_function(
        Eigen::Map<const Eigen::VectorXd>(alphas.data(), L),
        Eigen::Map<const Eigen::VectorXd>(colors.data(), 3 * L),
        kernels, 0.0, false);
}

double final_constraint_norm(const std::vector<double>& x, int L, const Eigen::Vector3d& target) {
    double cons[4];
    seq_constraint(x.data(), L, target, cons);
    return std::sqrt(cons[0] * cons[0] + cons[1] * cons[1] + cons[2] * cons[2] + cons[3] * cons[3]);
}

}

int main() {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> cd(0.0, 1.0);
    const int L = 4;
    const int dim = 4 * L;
    const int cases = 50;

    std::vector<ColorKernel> kernels(L);
    for (int i = 0; i < L; ++i) {
        kernels[i].mu = Eigen::Vector3d(cd(rng), cd(rng), cd(rng));
        Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * 0.01;
        cov(0, 1) = cov(1, 0) = 0.002;
        kernels[i].sigma_inv = cov.inverse();
    }

    struct CaseResult {
        int c;
        double x_diff;
        double e_diff;
        double c_diff;
        std::vector<double> proto;
        std::vector<double> seq;
    };
    std::vector<CaseResult> results;
    results.reserve(cases);

    for (int c = 0; c < cases; ++c) {
        SolverParams sp;
        sp.target_color = Eigen::Vector3d(cd(rng), cd(rng), cd(rng));
        sp.use_sparsity = false;
        sp.use_target_alphas = false;

        const Eigen::VectorXd proto = solve_per_pixel(kernels, sp);
        std::vector<double> proto_v(dim);
        for (int i = 0; i < dim; ++i) proto_v[i] = proto(i);

        const std::vector<double> seq_v = seq_solve_per_pixel(kernels, sp.target_color);

        const double x_diff = max_abs_diff(proto_v, seq_v);
        const double e_diff = std::fabs(final_energy(proto_v, L, kernels) - final_energy(seq_v, L, kernels));
        const double c_diff = std::fabs(final_constraint_norm(proto_v, L, sp.target_color)
                                       - final_constraint_norm(seq_v, L, sp.target_color));
        results.push_back({c, x_diff, e_diff, c_diff, proto_v, seq_v});
    }

    std::vector<CaseResult> sorted = results;
    std::sort(sorted.begin(), sorted.end(), [](const CaseResult& a, const CaseResult& b) {
        return a.x_diff > b.x_diff;
    });

    double max_e_diff = 0.0; int max_e_case = -1;
    double max_c_diff = 0.0; int max_c_case = -1;
    for (const auto& r : results) {
        if (r.e_diff > max_e_diff) { max_e_diff = r.e_diff; max_e_case = r.c; }
        if (r.c_diff > max_c_diff) { max_c_diff = r.c_diff; max_c_case = r.c; }
    }

    std::printf("=== L2c summary (%d cases) ===\n", cases);
    std::printf("max x_diff:  %.3e (case %d)\n", sorted[0].x_diff, sorted[0].c);
    std::printf("max e_diff:  %.3e (case %d)\n", max_e_diff, max_e_case);
    std::printf("max c_diff:  %.3e (case %d)\n", max_c_diff, max_c_case);

    const int n_top = std::min(5, static_cast<int>(sorted.size()));
    std::printf("\nTop %d worst cases (by x_diff):\n", n_top);
    for (int i = 0; i < n_top; ++i) {
        const auto& r = sorted[i];
        std::printf("\ncase %d: x_diff=%.3e  e_diff=%.3e  c_diff=%.3e\n",
                    r.c, r.x_diff, r.e_diff, r.c_diff);
        std::printf("  proto x: ");
        for (int k = 0; k < dim; ++k) std::printf("%.6f ", r.proto[k]);
        std::printf("\n");
        std::printf("  seq   x: ");
        for (int k = 0; k < dim; ++k) std::printf("%.6f ", r.seq[k]);
        std::printf("\n");
    }
    return 0;
}