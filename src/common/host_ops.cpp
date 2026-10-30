#include "host_ops.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

namespace unmixing {
namespace host {

namespace {

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

GrayImage convolve3x3(const GrayImage& img,
                      double k00, double k01, double k02,
                      double k10, double k11, double k12,
                      double k20, double k21, double k22) {
    const int W = img.width(), H = img.height();
    GrayImage out(W, H);
    const double k[3][3] = {
        { k00, k01, k02 },
        { k10, k11, k12 },
        { k20, k21, k22 }
    };
    for (int y = 0; y < H; ++y) {
        double* o = out.row(y);
        for (int x = 0; x < W; ++x) {
            double s = 0.0;
            for (int ky = 0; ky < 3; ++ky) {
                const double* r = img.row(clampi(y + ky - 1, 0, H - 1));
                for (int kx = 0; kx < 3; ++kx) {
                    s += k[ky][kx] * r[clampi(x + kx - 1, 0, W - 1)];
                }
            }
            o[x] = s;
        }
    }
    return out;
}

}

GrayImage luminance(const ColorImage& img) {
    const int W = img.width(), H = img.height();
    GrayImage out(W, H);
    for (int y = 0; y < H; ++y) {
        const double* r = img.r().row(y);
        const double* g = img.g().row(y);
        const double* b = img.b().row(y);
        double* o = out.row(y);
        for (int x = 0; x < W; ++x)
            o[x] = 0.2126 * r[x] + 0.7152 * g[x] + 0.0722 * b[x];
    }
    return out;
}

GrayImage sobel_x(const GrayImage& img) {
    return convolve3x3(img, +1, 0, -1, +2, 0, -2, +1, 0, -1);
}
GrayImage sobel_y(const GrayImage& img) {
    return convolve3x3(img, +1, +2, +1, 0, 0, 0, -1, -2, -1);
}

GrayImage gradient_magnitude(const GrayImage& img) {
    const GrayImage gx = sobel_x(img);
    const GrayImage gy = sobel_y(img);
    const int W = img.width(), H = img.height();
    GrayImage out(W, H);
    for (int y = 0; y < H; ++y) {
        const double* rx = gx.row(y);
        const double* ry = gy.row(y);
        double* o = out.row(y);
        for (int x = 0; x < W; ++x) o[x] = std::sqrt(rx[x] * rx[x] + ry[x] * ry[x]);
    }
    return out;
}

GrayImage box_filter(const GrayImage& img, int radius) {
    if (radius <= 0) return img;
    const int W = img.width(), H = img.height();
    const double inv = 1.0 / (2 * radius + 1);

    GrayImage tmp(W, H);
    for (int y = 0; y < H; ++y) {
        const double* r = img.row(y);
        double* o = tmp.row(y);
        double s = 0.0;
        for (int k = -radius; k <= radius; ++k) s += r[clampi(k, 0, W - 1)];
        for (int x = 0; x < W; ++x) {
            o[x] = s * inv;
            s -= r[clampi(x - radius, 0, W - 1)];
            s += r[clampi(x + radius + 1, 0, W - 1)];
        }
    }

    GrayImage out(W, H);
    std::vector<double> col(H);
    for (int x = 0; x < W; ++x) {
        double s = 0.0;
        for (int k = -radius; k <= radius; ++k) s += tmp.at(x, clampi(k, 0, H - 1));
        for (int y = 0; y < H; ++y) {
            out.at(x, y) = s * inv;
            s -= tmp.at(x, clampi(y - radius, 0, H - 1));
            s += tmp.at(x, clampi(y + radius + 1, 0, H - 1));
        }
    }
    return out;
}

GrayImage multiply(const GrayImage& a, const GrayImage& b) {
    const int W = a.width(), H = a.height();
    GrayImage out(W, H);
    for (int y = 0; y < H; ++y) {
        const double* ra = a.row(y);
        const double* rb = b.row(y);
        double* o = out.row(y);
        for (int x = 0; x < W; ++x) o[x] = ra[x] * rb[x];
    }
    return out;
}

GrayImage square(const GrayImage& a) { return multiply(a, a); }

GrayImage subtract(const GrayImage& a, const GrayImage& b) {
    const int W = a.width(), H = a.height();
    GrayImage out(W, H);
    for (int y = 0; y < H; ++y) {
        const double* ra = a.row(y);
        const double* rb = b.row(y);
        double* o = out.row(y);
        for (int x = 0; x < W; ++x) o[x] = ra[x] - rb[x];
    }
    return out;
}

GrayImage add(const GrayImage& a, const GrayImage& b) {
    const int W = a.width(), H = a.height();
    GrayImage out(W, H);
    for (int y = 0; y < H; ++y) {
        const double* ra = a.row(y);
        const double* rb = b.row(y);
        double* o = out.row(y);
        for (int x = 0; x < W; ++x) o[x] = ra[x] + rb[x];
    }
    return out;
}

void force_unity(GrayImage& img) {
    const std::size_t n = static_cast<std::size_t>(img.width()) * img.height();
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += img.data()[i];
    if (s > 1e-16) {
        const double inv = 1.0 / s;
        for (std::size_t i = 0; i < n; ++i) img.data()[i] *= inv;
    }
}

void scale_to_unit(GrayImage& img) {
    const std::size_t n = static_cast<std::size_t>(img.width()) * img.height();
    double mn = DBL_MAX, mx = -DBL_MAX;
    for (std::size_t i = 0; i < n; ++i) {
        mn = std::min(mn, img.data()[i]);
        mx = std::max(mx, img.data()[i]);
    }
    const double range = mx - mn;
    if (range > 0.0) {
        const double inv = 1.0 / range;
        for (std::size_t i = 0; i < n; ++i) img.data()[i] = (img.data()[i] - mn) * inv;
    }
}

GrayImage guided_filter(const GrayImage& input,
                        const GrayImage& guide_r,
                        const GrayImage& guide_g,
                        const GrayImage& guide_b,
                        int radius, double epsilon) {
    const int W = input.width(), H = input.height();
    const GrayImage mean_r  = box_filter(guide_r, radius);
    const GrayImage mean_g  = box_filter(guide_g, radius);
    const GrayImage mean_b  = box_filter(guide_b, radius);
    const GrayImage mean_p  = box_filter(input, radius);
    const GrayImage mean_rp = box_filter(multiply(guide_r, input), radius);
    const GrayImage mean_gp = box_filter(multiply(guide_g, input), radius);
    const GrayImage mean_bp = box_filter(multiply(guide_b, input), radius);
    const GrayImage cov_rp  = subtract(mean_rp, multiply(mean_r, mean_p));
    const GrayImage cov_gp  = subtract(mean_gp, multiply(mean_g, mean_p));
    const GrayImage cov_bp  = subtract(mean_bp, multiply(mean_b, mean_p));
    const GrayImage var_rr  = subtract(box_filter(square(guide_r), radius), square(mean_r));
    const GrayImage var_rg  = subtract(box_filter(multiply(guide_r, guide_g), radius), multiply(mean_r, mean_g));
    const GrayImage var_rb  = subtract(box_filter(multiply(guide_r, guide_b), radius), multiply(mean_r, mean_b));
    const GrayImage var_gg  = subtract(box_filter(square(guide_g), radius), square(mean_g));
    const GrayImage var_gb  = subtract(box_filter(multiply(guide_g, guide_b), radius), multiply(mean_g, mean_b));
    const GrayImage var_bb  = subtract(box_filter(square(guide_b), radius), square(mean_b));

    GrayImage a_r(W, H), a_g(W, H), a_b(W, H);
    const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            Eigen::Matrix3d S;
            S << var_rr.at(x, y), var_rg.at(x, y), var_rb.at(x, y),
                 var_rg.at(x, y), var_gg.at(x, y), var_gb.at(x, y),
                 var_rb.at(x, y), var_gb.at(x, y), var_bb.at(x, y);
            const Eigen::Vector3d cov(cov_rp.at(x, y), cov_gp.at(x, y), cov_bp.at(x, y));
            const Eigen::Vector3d a = (S + epsilon * I3).inverse() * cov;
            a_r.at(x, y) = a(0);
            a_g.at(x, y) = a(1);
            a_b.at(x, y) = a(2);
        }
    }
    const GrayImage b = subtract(subtract(subtract(mean_p, multiply(a_r, mean_r)),
                                          multiply(a_g, mean_g)), multiply(a_b, mean_b));
    GrayImage q = box_filter(b, radius);
    q = add(q, multiply(box_filter(a_r, radius), guide_r));
    q = add(q, multiply(box_filter(a_g, radius), guide_g));
    q = add(q, multiply(box_filter(a_b, radius), guide_b));
    return q;
}

GrayImage guided_filter_kernel(const GrayImage& img,
                               int cx, int cy, int radius,
                               double epsilon, bool force_positive) {
    const int W = img.width(), H = img.height();
    const GrayImage mean_I = box_filter(img, radius);
    const GrayImage corr_I = box_filter(square(img), radius);
    const GrayImage var_I  = subtract(corr_I, square(mean_I));
    const double I_seed = img.at(cx, cy);

    const int Wp = W + 1;
    const std::size_t stride = static_cast<std::size_t>(Wp);
    std::vector<double> IA(static_cast<std::size_t>(H + 1) * stride, 0.0);
    std::vector<double> IC(static_cast<std::size_t>(H + 1) * stride, 0.0);
    std::vector<double> I1(static_cast<std::size_t>(H + 1) * stride, 0.0);

    for (int j = 0; j < H; ++j) {
        double rowA = 0.0, rowC = 0.0, rowOne = 0.0;
        for (int i = 0; i < W; ++i) {
            const double mu    = mean_I.at(i, j);
            const double var   = var_I.at(i, j);
            const double denom = epsilon + var;
            const double A = (I_seed - mu) / denom;
            const double C = (I_seed - mu) * mu / denom;
            rowA   += A;
            rowC   += C;
            rowOne += 1.0;
            IA[(static_cast<std::size_t>(j + 1)) * stride + (i + 1)] = IA[(static_cast<std::size_t>(j)) * stride + (i + 1)] + rowA;
            IC[(static_cast<std::size_t>(j + 1)) * stride + (i + 1)] = IC[(static_cast<std::size_t>(j)) * stride + (i + 1)] + rowC;
            I1[(static_cast<std::size_t>(j + 1)) * stride + (i + 1)] = I1[(static_cast<std::size_t>(j)) * stride + (i + 1)] + rowOne;
        }
    }

    auto rect_sum = [stride](const std::vector<double>& M, int x0, int y0, int x1, int y1) -> double {
        const std::size_t r0 = static_cast<std::size_t>(y0) * stride;
        const std::size_t r1 = static_cast<std::size_t>(y1 + 1) * stride;
        return M[r1 + (x1 + 1)] - M[r1 + x0] - M[r0 + (x1 + 1)] + M[r0 + x0];
    };

    GrayImage weight(W, H, 0.0);
    for (int y = cy - radius; y <= cy + radius; ++y) {
        if (y < 0 || y >= H) continue;
        for (int x = cx - radius; x <= cx + radius; ++x) {
            if (x < 0 || x >= W) continue;
            int kx0 = std::max(x - radius, cx - radius);
            int kx1 = std::min(x + radius, cx + radius);
            int ky0 = std::max(y - radius, cy - radius);
            int ky1 = std::min(y + radius, cy + radius);
            kx0 = std::max(kx0, 0); kx1 = std::min(kx1, W - 1);
            ky0 = std::max(ky0, 0); ky1 = std::min(ky1, H - 1);
            if (kx0 > kx1 || ky0 > ky1) continue;
            const double s1 = rect_sum(I1, kx0, ky0, kx1, ky1);
            const double sA = rect_sum(IA, kx0, ky0, kx1, ky1);
            const double sC = rect_sum(IC, kx0, ky0, kx1, ky1);
            const double I_j = img.at(x, y);
            double w = s1 + I_j * sA - sC;
            if (force_positive && w < 0.0) w = 0.0;
            weight.at(x, y) = w;
        }
    }
    force_unity(weight);
    return weight;
}

void compute_normal_distribution(const ColorImage& img, const GrayImage& weight,
                                 Eigen::Vector3d& mu, Eigen::Matrix3d& sigma) {
    const int W = img.width(), H = img.height();
    mu.setZero();
    for (int y = 0; y < H; ++y) {
        const double* w = weight.row(y);
        const double* r = img.r().row(y);
        const double* g = img.g().row(y);
        const double* b = img.b().row(y);
        for (int x = 0; x < W; ++x) {
            const Eigen::Vector3d c(r[x], g[x], b[x]);
            mu += w[x] * c;
        }
    }
    sigma.setZero();
    for (int y = 0; y < H; ++y) {
        const double* w = weight.row(y);
        const double* r = img.r().row(y);
        const double* g = img.g().row(y);
        const double* b = img.b().row(y);
        for (int x = 0; x < W; ++x) {
            const Eigen::Vector3d c(r[x], g[x], b[x]);
            const Eigen::Vector3d d = c - mu;
            sigma += w[x] * d * d.transpose();
        }
    }
    sigma += params::kNormalDistEpsilon * Eigen::Matrix3d::Identity();
}

}
}