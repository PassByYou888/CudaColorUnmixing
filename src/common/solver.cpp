#include "solver.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

namespace unmixing {

namespace {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

Vector3d composite_color(const VectorXd& alphas, const VectorXd& colors) {
    const int L = static_cast<int>(alphas.rows());
    Vector3d sc = Vector3d::Zero();
    for (int i = 0; i < L; ++i) sc += alphas(i) * colors.segment<3>(i * 3);
    return sc;
}

Vector4d constraint_vector(const VectorXd& x,
                           const Vector3d& target,
                           bool use_target_alphas,
                           const VectorXd& target_alphas) {
    const int L = static_cast<int>(x.rows()) / 4;
    const VectorXd alphas = x.head(L);
    const VectorXd colors = x.tail(3 * L);
    const Vector3d sc = composite_color(alphas, colors);
    const Vector3d gc = (sc - target).cwiseProduct(sc - target);
    const double sa = alphas.sum();
    const double ga = use_target_alphas
        ? (alphas - target_alphas).squaredNorm()
        : (sa - 1.0) * (sa - 1.0);
    return Vector4d(gc(0), gc(1), gc(2), ga);
}

double energy(const VectorXd& alphas, const VectorXd& colors,
              const std::vector<ColorKernel>& kernels,
              double sigma, bool use_sparsity) {
    const int L = static_cast<int>(alphas.rows());
    double e = 0.0;
    for (int i = 0; i < L; ++i)
        e += alphas(i) * kernels[i].squared_mahalanobis(colors.segment<3>(i * 3));
    if (use_sparsity) {
        const double sn = alphas.squaredNorm();
        if (sn > 1e-30) e += sigma * ((alphas.sum() / sn) - 1.0);
    }
    if (!std::isfinite(e)) e = 1e10;
    return e;
}

VectorXd grad_energy(const VectorXd& alphas, const VectorXd& colors,
                     const std::vector<ColorKernel>& kernels,
                     double sigma, bool use_sparsity) {
    const int L = static_cast<int>(alphas.rows());
    VectorXd g = VectorXd::Zero(4 * L);
    for (int i = 0; i < L; ++i) {
        const ColorKernel& k = kernels[i];
        const Vector3d u = colors.segment<3>(i * 3);
        g(i) = k.squared_mahalanobis(u);
        g.segment<3>(L + i * 3) = 2.0 * alphas(i) * (k.sigma_inv * (u - k.mu));
    }
    if (use_sparsity) {
        const double sa  = alphas.sum();
        const double sa2 = alphas.squaredNorm();
        if (sa2 > 1e-15) {
            const double denom = sa2 * sa2;
            for (int i = 0; i < L; ++i)
                g(i) += sigma * (sa2 - 2.0 * alphas(i) * sa) / denom;
        }
    }
    for (int i = 0; i < 4 * L; ++i)
        if (!std::isfinite(g(i))) g(i) = 0.0;
    return g;
}

double constraint_terms(const Vector4d& c, const Vector4d& lambda, double lo) {
    return lambda.dot(c) + 0.5 * lo * c.squaredNorm();
}

VectorXd grad_constraint_terms(const VectorXd& alphas, const VectorXd& colors,
                               const Vector3d& target, const Vector4d& c,
                               const Vector4d& lambda, double lo,
                               bool use_target_alphas, const VectorXd& target_alphas) {
    const int L = static_cast<int>(alphas.rows());
    VectorXd g = VectorXd::Zero(4 * L);
    const double sa = alphas.sum();
    const Vector3d sc = composite_color(alphas, colors);
    for (int i = 0; i < L; ++i) {
        const Vector3d u = colors.segment<3>(i * 3);
        const double a = alphas(i);
        const Vector3d dgc_da = 2.0 * u.cwiseProduct(sc - target);
        const double dga_da = use_target_alphas
            ? 2.0 * (a - target_alphas(i))
            : 2.0 * (sa - 1.0);
        const Vector4d dg_da(dgc_da(0), dgc_da(1), dgc_da(2), dga_da);
        const double dgr = 2.0 * a * (sc(0) - target(0));
        const double dgg = 2.0 * a * (sc(1) - target(1));
        const double dgb = 2.0 * a * (sc(2) - target(2));
        g(i)               = lambda.head<3>().dot(dgc_da) + lambda(3) * dga_da + lo * c.dot(dg_da);
        g(L + i * 3 + 0)   = lambda(0) * dgr + lo * c(0) * dgr;
        g(L + i * 3 + 1)   = lambda(1) * dgg + lo * c(1) * dgg;
        g(L + i * 3 + 2)   = lambda(2) * dgb + lo * c(2) * dgb;
    }
    for (int i = 0; i < 4 * L; ++i)
        if (!std::isfinite(g(i))) g(i) = 0.0;
    return g;
}

VectorXd lbfgs_minimize(const VectorXd& x0,
                        const std::function<double(const VectorXd&, VectorXd&)>& fg,
                        const VectorXd& lower, const VectorXd& upper,
                        int max_iter, int m) {
    const int n = static_cast<int>(x0.rows());
    VectorXd x = x0;
    std::vector<VectorXd> S(m), Y(m);
    std::vector<double> rho(m, 0.0);
    int k = 0;
    VectorXd g(n);
    double f = fg(x, g);

    for (int iter = 0; iter < max_iter; ++iter) {
        VectorXd pg = g;
        for (int i = 0; i < n; ++i) {
            if ((x(i) <= lower(i) && pg(i) > 0.0) || (x(i) >= upper(i) && pg(i) < 0.0))
                pg(i) = 0.0;
        }
        if (pg.norm() < 1e-10) break;

        const int m_used = std::min(k, m);
        VectorXd q = pg;
        std::vector<double> alpha(m_used, 0.0);
        for (int j = m_used - 1; j >= 0; --j) {
            const int idx = (k - m_used + j) % m;
            alpha[j] = rho[idx] * S[idx].dot(q);
            q -= alpha[j] * Y[idx];
        }
        double gamma = 1.0;
        if (m_used > 0) {
            const int idx_last = (k - 1) % m;
            const double sy = S[idx_last].dot(Y[idx_last]);
            if (sy > 1e-10) gamma = sy / Y[idx_last].squaredNorm();
        }
        VectorXd d = gamma * q;
        for (int j = 0; j < m_used; ++j) {
            const int idx = (k - m_used + j) % m;
            const double beta = rho[idx] * Y[idx].dot(d);
            d += (alpha[j] - beta) * S[idx];
        }
        d = -d;

        double dphi0 = g.dot(d);
        if (!std::isfinite(dphi0) || dphi0 >= 0.0) {
            k = 0;
            d = -pg;
            dphi0 = -pg.squaredNorm();
        }

        double step = 1.0;
        VectorXd x_new(n), g_new(n);
        double f_new = f;
        bool accepted = false;
        for (int ls = 0; ls < 30; ++ls) {
            x_new = x + step * d;
            for (int i = 0; i < n; ++i)
                x_new(i) = std::clamp(x_new(i), lower(i), upper(i));
            f_new = fg(x_new, g_new);
            if (std::isfinite(f_new) && f_new <= f + 1e-4 * g.dot(x_new - x)) {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if (!accepted) {
            x_new = x;
            g_new = g;
            f_new = f;
        }

        VectorXd s = x_new - x;
        VectorXd y = g_new - g;
        const double sy = s.dot(y);

        if (sy > 1e-10 && std::isfinite(sy)) {
            const int slot = k % m;
            S[slot] = s;
            Y[slot] = y;
            rho[slot] = 1.0 / sy;
            ++k;
        } else {
            k = 0;
        }

        const double dx = s.norm();
        const double df = std::abs(f - f_new);
        x = x_new;
        g = g_new;
        f = f_new;
        if (dx < 1e-8 || df < 1e-12) break;
    }
    return x;
}

VectorXd build_initial(const std::vector<ColorKernel>& kernels, const SolverParams& sp) {
    const int L = static_cast<int>(kernels.size());
    VectorXd x = VectorXd::Zero(4 * L);
    if (!sp.use_target_alphas) {
        double md = DBL_MAX;
        int closest = 0;
        for (int i = 0; i < L; ++i) {
            const double d = kernels[i].squared_mahalanobis(sp.target_color);
            if (d < md) { md = d; closest = i; }
        }
        x(closest) = 1.0;
        for (int i = 0; i < L; ++i) {
            const Vector3d c = (i == closest) ? sp.target_color : kernels[i].mu;
            for (int j = 0; j < 3; ++j)
                x(L + i * 3 + j) = std::clamp(c(j), 0.0, 1.0);
        }
    } else {
        if ((int)sp.target_alphas.size() != L || (int)sp.initial_colors.size() != 3 * L) {
            std::fprintf(stderr, "[solver] build_initial: target_alphas=%lld initial_colors=%lld expected %d/%d\n",
                         (long long)sp.target_alphas.size(),
                         (long long)sp.initial_colors.size(), L, 3 * L);
            return VectorXd();
        }
        x.head(L)     = sp.target_alphas;
        x.tail(3 * L) = sp.initial_colors;
    }
    return x;
}

}

double energy_function(const VectorXd& alphas, const VectorXd& colors,
                       const std::vector<ColorKernel>& kernels,
                       double sigma, bool use_sparsity) {
    return energy(alphas, colors, kernels, sigma, use_sparsity);
}

VectorXd solve_per_pixel(const std::vector<ColorKernel>& kernels, const SolverParams& sp) {
    const int L = static_cast<int>(kernels.size());
    if (L == 0) return VectorXd();
    const int n = 4 * L;

    const VectorXd lower = VectorXd::Zero(n);
    const VectorXd upper = VectorXd::Ones(n);
    double lo = sp.lo_init;
    Vector4d lambda = Vector4d::Constant(sp.lambda_init);
    VectorXd x = build_initial(kernels, sp);
    if (x.size() != n) {
        std::fprintf(stderr, "[solver] solve_per_pixel: build_initial failed\n");
        return VectorXd();
    }
    Vector4d g_old = constraint_vector(x, sp.target_color, sp.use_target_alphas, sp.target_alphas);

    for (int outer = 0; outer < params::kMaxALMIter; ++outer) {
        auto fg = [&](const VectorXd& x_, VectorXd& grad) -> double {
            const VectorXd alphas = x_.head(L);
            const VectorXd colors = x_.tail(3 * L);
            const Vector4d c = constraint_vector(x_, sp.target_color, sp.use_target_alphas, sp.target_alphas);
            const double e = energy(alphas, colors, kernels, sp.sigma, sp.use_sparsity);
            const double p = constraint_terms(c, lambda, lo);
            if (grad.size() == n) {
                grad = grad_energy(alphas, colors, kernels, sp.sigma, sp.use_sparsity)
                     + grad_constraint_terms(alphas, colors, sp.target_color, c, lambda, lo,
                                             sp.use_target_alphas, sp.target_alphas);
            }
            return e + p;
        };

        VectorXd x_new = lbfgs_minimize(x, fg, lower, upper, params::kMaxInnerEval, 6);
        const Vector4d g_new = constraint_vector(x_new, sp.target_color, sp.use_target_alphas, sp.target_alphas);
        lambda += lo * g_new;
        if (!std::isfinite(lambda.norm())) lambda = Vector4d::Constant(sp.lambda_init);
        if (!std::isfinite(g_new.norm()) || g_new.norm() > params::kALMGamma * g_old.norm()) lo *= params::kALMBeta;
        if (!std::isfinite(lo) || lo > 1e6) lo = 1e6;

        const bool unchanged = (x_new - x).squaredNorm() < params::kALMEpsilon;
        const bool satisfied = g_new.norm() < params::kALMEpsilon;
        x = x_new;
        g_old = g_new;
        if (unchanged && satisfied) break;
    }

    for (int i = 0; i < L; ++i) {
        double a = std::clamp(x(i), 0.0, 1.0);
        x(i) = a;
        if (a < 1e-4) {
            x(L + i*3 + 0) = kernels[i].mu(0);
            x(L + i*3 + 1) = kernels[i].mu(1);
            x(L + i*3 + 2) = kernels[i].mu(2);
        } else {
            x(L + i*3 + 0) = std::clamp(x(L + i*3 + 0), 0.0, 1.0);
            x(L + i*3 + 1) = std::clamp(x(L + i*3 + 1), 0.0, 1.0);
            x(L + i*3 + 2) = std::clamp(x(L + i*3 + 2), 0.0, 1.0);
        }
    }
    return x;
}

}