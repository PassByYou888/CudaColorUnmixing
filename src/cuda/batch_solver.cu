#include "batch_solver.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

#define UNMIX_WARP_SIZE        32
#define UNMIX_ALM_GAMMA         0.25
#define UNMIX_ALM_BETA          10.0
#define UNMIX_ARMijo_C1         1.0e-4
#define UNMIX_LS_MAX_BACKTRACK  30
#define UNMIX_LBFGS_M           6
#define UNMIX_INNER_MAX_ITER    200
#define UNMIX_INNER_DX_TOL      1e-8
#define UNMIX_INNER_DF_TOL      1e-12
#define UNMIX_PG_TOL            1e-10
#define UNMIX_LO_MAX            1e6
#define UNMIX_SPARSITY_ENERGY_EPS 1e-30
#define UNMIX_SPARSITY_GRAD_EPS   1e-15

namespace unmixing {
namespace cuda {

namespace {

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "[batch_solver] %s failed: %s (line %d)\n", \
                     #call, cudaGetErrorString(_e), __LINE__); \
        std::abort(); \
    } \
} while (0)

__device__ __forceinline__ bool is_finite(double v) {
    return fabs(v) <= 1e308;
}

__device__ __forceinline__ double warp_sum_broadcast(double val) {
    #pragma unroll
    for (int offset = UNMIX_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ void compute_summary_warp(
    const double* __restrict__ x_d, int L,
    double* __restrict__ sc_out, double* __restrict__ sa_out, int lane)
{
    double sc0 = 0.0, sc1 = 0.0, sc2 = 0.0, sa = 0.0;
    for (int i = lane; i < L; i += UNMIX_WARP_SIZE) {
        const double a = x_d[i];
        sc0 += a * x_d[L + i * 3 + 0];
        sc1 += a * x_d[L + i * 3 + 1];
        sc2 += a * x_d[L + i * 3 + 2];
        sa  += a;
    }
    sc_out[0] = warp_sum_broadcast(sc0);
    sc_out[1] = warp_sum_broadcast(sc1);
    sc_out[2] = warp_sum_broadcast(sc2);
    sa_out[0] = warp_sum_broadcast(sa);
}

__device__ void compute_constraint_warp(
    const double* __restrict__ x_d, int L,
    const double* __restrict__ target_color,
    const double* __restrict__ target_alphas,
    int use_target_alphas,
    double* __restrict__ cons, int lane)
{
    double sc[3];
    double sa;
    compute_summary_warp(x_d, L, sc, &sa, lane);

    cons[0] = (sc[0] - target_color[0]) * (sc[0] - target_color[0]);
    cons[1] = (sc[1] - target_color[1]) * (sc[1] - target_color[1]);
    cons[2] = (sc[2] - target_color[2]) * (sc[2] - target_color[2]);

    if (use_target_alphas) {
        double d = 0.0;
        for (int i = lane; i < L; i += UNMIX_WARP_SIZE) {
            const double diff = x_d[i] - target_alphas[i];
            d += diff * diff;
        }
        cons[3] = warp_sum_broadcast(d);
    } else {
        cons[3] = (sa - 1.0) * (sa - 1.0);
    }
}

__device__ double eval_fg_warp(
    const double* __restrict__ x_d,
    double* __restrict__ g_d,
    int L, int dim,
    const double* __restrict__ kernel_mu,
    const double* __restrict__ kernel_sigma_inv,
    const double* __restrict__ target_color,
    const double* __restrict__ target_alphas,
    int use_sparsity, int use_target_alphas,
    double sigma_sparsity,
    const double* __restrict__ lambda,
    double lo,
    int lane)
{
    double sc0 = 0.0, sc1 = 0.0, sc2 = 0.0, sa = 0.0, energy = 0.0;
    for (int i = lane; i < L; i += UNMIX_WARP_SIZE) {
        const double a = x_d[i];
        sc0 += a * x_d[L + i * 3 + 0];
        sc1 += a * x_d[L + i * 3 + 1];
        sc2 += a * x_d[L + i * 3 + 2];
        sa  += a;
        const double* col = x_d + L + i * 3;
        const double* mu  = kernel_mu + i * 3;
        const double dr = col[0] - mu[0];
        const double dg = col[1] - mu[1];
        const double db = col[2] - mu[2];
        const double* s = kernel_sigma_inv + i * 9;
        const double maha =
            dr * (s[0] * dr + s[1] * dg + s[2] * db) +
            dg * (s[3] * dr + s[4] * dg + s[5] * db) +
            db * (s[6] * dr + s[7] * dg + s[8] * db);
        energy += a * maha;
    }
    #pragma unroll
    for (int offset = UNMIX_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sc0    += __shfl_xor_sync(0xffffffffu, sc0, offset);
        sc1    += __shfl_xor_sync(0xffffffffu, sc1, offset);
        sc2    += __shfl_xor_sync(0xffffffffu, sc2, offset);
        sa     += __shfl_xor_sync(0xffffffffu, sa, offset);
        energy += __shfl_xor_sync(0xffffffffu, energy, offset);
    }

    double cons[4];
    cons[0] = (sc0 - target_color[0]) * (sc0 - target_color[0]);
    cons[1] = (sc1 - target_color[1]) * (sc1 - target_color[1]);
    cons[2] = (sc2 - target_color[2]) * (sc2 - target_color[2]);
    if (use_target_alphas) {
        double d = 0.0;
        for (int i = lane; i < L; i += UNMIX_WARP_SIZE) {
            const double diff = x_d[i] - target_alphas[i];
            d += diff * diff;
        }
        cons[3] = warp_sum_broadcast(d);
    } else {
        cons[3] = (sa - 1.0) * (sa - 1.0);
    }

    double sa2 = 0.0;
    if (use_sparsity) {
        for (int i = lane; i < L; i += UNMIX_WARP_SIZE) sa2 += x_d[i] * x_d[i];
        sa2 = warp_sum_broadcast(sa2);
        if (sa2 > UNMIX_SPARSITY_ENERGY_EPS) energy += sigma_sparsity * (sa / sa2 - 1.0);
    }

    double penalty = 0.0;
    for (int j = 0; j < 4; ++j) {
        penalty += lambda[j] * cons[j] + 0.5 * lo * cons[j] * cons[j];
    }
    double f = energy + penalty;
    if (!is_finite(f)) f = 1e10;

    for (int i = lane; i < L; i += UNMIX_WARP_SIZE) {
        const double a = x_d[i];
        const double* col = x_d + L + i * 3;
        const double* mu  = kernel_mu + i * 3;
        const double dr = col[0] - mu[0];
        const double dg = col[1] - mu[1];
        const double db = col[2] - mu[2];
        const double* s = kernel_sigma_inv + i * 9;
        const double maha =
            dr * (s[0] * dr + s[1] * dg + s[2] * db) +
            dg * (s[3] * dr + s[4] * dg + s[5] * db) +
            db * (s[6] * dr + s[7] * dg + s[8] * db);
        g_d[i] = maha;
        g_d[L + i * 3 + 0] = 2.0 * a * (s[0] * dr + s[1] * dg + s[2] * db);
        g_d[L + i * 3 + 1] = 2.0 * a * (s[3] * dr + s[4] * dg + s[5] * db);
        g_d[L + i * 3 + 2] = 2.0 * a * (s[6] * dr + s[7] * dg + s[8] * db);
    }

    if (use_sparsity && sa2 > UNMIX_SPARSITY_GRAD_EPS) {
        for (int i = lane; i < L; i += UNMIX_WARP_SIZE) {
            g_d[i] += sigma_sparsity * (sa2 - 2.0 * x_d[i] * sa) / (sa2 * sa2);
        }
    }

    const double sc_arr[3] = { sc0, sc1, sc2 };
    for (int j = 0; j < 3; ++j) {
        const double dpen = lambda[j] + lo * cons[j];
        const double factor = 2.0 * (sc_arr[j] - target_color[j]) * dpen;
        for (int i = lane; i < L; i += UNMIX_WARP_SIZE) {
            g_d[i] += factor * x_d[L + i * 3 + j];
            g_d[L + i * 3 + j] += factor * x_d[i];
        }
    }
    if (use_target_alphas) {
        const double dpen = lambda[3] + lo * cons[3];
        for (int i = lane; i < L; i += UNMIX_WARP_SIZE) {
            g_d[i] += 2.0 * dpen * (x_d[i] - target_alphas[i]);
        }
    } else {
        const double dpen = lambda[3] + lo * cons[3];
        const double factor = 2.0 * (sa - 1.0) * dpen;
        for (int i = lane; i < L; i += UNMIX_WARP_SIZE) {
            g_d[i] += factor;
        }
    }

    for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) {
        if (!is_finite(g_d[i])) g_d[i] = 0.0;
    }

    return f;
}

__device__ void lbfgs_minimize_warp(
    double* __restrict__ x_d,
    double* __restrict__ g_d,
    int L, int dim,
    const double* __restrict__ kernel_mu,
    const double* __restrict__ kernel_sigma_inv,
    const double* __restrict__ target_color,
    const double* __restrict__ target_alphas,
    int use_sparsity, int use_target_alphas,
    double sigma_sparsity,
    const double* __restrict__ lambda,
    double lo,
    double* __restrict__ S_d,
    double* __restrict__ Y_d,
    double* __restrict__ rho_d,
    double* __restrict__ q_d,
    double* __restrict__ d_d,
    double* __restrict__ x_try_d,
    double* __restrict__ g_try_d,
    int max_iter,
    int lane)
{
    int k = 0;
    double f_cur = eval_fg_warp(x_d, g_d, L, dim,
                                kernel_mu, kernel_sigma_inv,
                                target_color, target_alphas,
                                use_sparsity, use_target_alphas,
                                sigma_sparsity, lambda, lo, lane);
    __syncwarp();

    for (int iter = 0; iter < max_iter; ++iter) {
        double pg_norm_sq = 0.0;
        for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) {
            double gi = g_d[i];
            if ((x_d[i] <= 0.0 && gi > 0.0) || (x_d[i] >= 1.0 && gi < 0.0)) gi = 0.0;
            q_d[i] = gi;
            pg_norm_sq += gi * gi;
        }
        pg_norm_sq = warp_sum_broadcast(pg_norm_sq);
        if (sqrt(pg_norm_sq) < UNMIX_PG_TOL) break;

        const int m_used = (k < UNMIX_LBFGS_M) ? k : UNMIX_LBFGS_M;

        for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) d_d[i] = q_d[i];
        __syncwarp();

        double alpha_local[UNMIX_LBFGS_M];
        #pragma unroll
        for (int j = 0; j < UNMIX_LBFGS_M; ++j) alpha_local[j] = 0.0;

        for (int j = m_used - 1; j >= 0; --j) {
            const int idx = (k - m_used + j) % UNMIX_LBFGS_M;
            const double* Sj = S_d + idx * dim;
            const double* Yj = Y_d + idx * dim;
            double sjq = 0.0;
            for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) sjq += Sj[i] * d_d[i];
            sjq = warp_sum_broadcast(sjq);
            const double rho_j = rho_d[idx];
            const double aj = rho_j * sjq;
            alpha_local[j] = aj;
            for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) d_d[i] -= aj * Yj[i];
            __syncwarp();
        }

        double gamma_h = 1.0;
        if (m_used > 0) {
            const int idx_last = (k - 1) % UNMIX_LBFGS_M;
            const double* Slast = S_d + idx_last * dim;
            const double* Ylast = Y_d + idx_last * dim;
            double sy = 0.0, yy = 0.0;
            for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) {
                sy += Slast[i] * Ylast[i];
                yy += Ylast[i] * Ylast[i];
            }
            sy = warp_sum_broadcast(sy);
            yy = warp_sum_broadcast(yy);
            if (sy > 1e-10) gamma_h = sy / yy;
        }
        for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) d_d[i] *= gamma_h;
        __syncwarp();

        for (int j = 0; j < m_used; ++j) {
            const int idx = (k - m_used + j) % UNMIX_LBFGS_M;
            const double* Sj = S_d + idx * dim;
            const double* Yj = Y_d + idx * dim;
            double yjd = 0.0;
            for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) yjd += Yj[i] * d_d[i];
            yjd = warp_sum_broadcast(yjd);
            const double rho_j = rho_d[idx];
            const double beta = rho_j * yjd;
            const double aj = alpha_local[j];
            for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) d_d[i] += (aj - beta) * Sj[i];
            __syncwarp();
        }

        for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) d_d[i] = -d_d[i];
        __syncwarp();

        double dphi0 = 0.0;
        for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) dphi0 += g_d[i] * d_d[i];
        dphi0 = warp_sum_broadcast(dphi0);

        if (!is_finite(dphi0) || dphi0 >= 0.0) {
            k = 0;
            for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) d_d[i] = -q_d[i];
            __syncwarp();
        }

        double step = 1.0;
        bool accepted = false;
        double f_new = f_cur;
        for (int ls = 0; ls < UNMIX_LS_MAX_BACKTRACK; ++ls) {
            for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) {
                x_try_d[i] = fmin(fmax(x_d[i] + step * d_d[i], 0.0), 1.0);
            }
            __syncwarp();

            double g_dot_diff = 0.0;
            for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) {
                g_dot_diff += g_d[i] * (x_try_d[i] - x_d[i]);
            }
            g_dot_diff = warp_sum_broadcast(g_dot_diff);

            f_new = eval_fg_warp(x_try_d, g_try_d, L, dim,
                                 kernel_mu, kernel_sigma_inv,
                                 target_color, target_alphas,
                                 use_sparsity, use_target_alphas,
                                 sigma_sparsity, lambda, lo, lane);
            __syncwarp();

            if (is_finite(f_new) && f_new <= f_cur + UNMIX_ARMijo_C1 * g_dot_diff) {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if (!accepted) {
            for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) {
                x_try_d[i] = x_d[i];
                g_try_d[i] = g_d[i];
            }
            f_new = f_cur;
            __syncwarp();
        }

        double dx_sq = 0.0;
        double sy = 0.0;
        for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) {
            const double dxi = x_try_d[i] - x_d[i];
            dx_sq += dxi * dxi;
            sy += dxi * (g_try_d[i] - g_d[i]);
        }
        dx_sq = warp_sum_broadcast(dx_sq);
        sy = warp_sum_broadcast(sy);
        const double dx = sqrt(dx_sq);
        const double df = fabs(f_cur - f_new);

        if (sy > 1e-10 && is_finite(sy)) {
            const int slot = k % UNMIX_LBFGS_M;
            double* Sslot = S_d + slot * dim;
            double* Yslot = Y_d + slot * dim;
            for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) {
                Sslot[i] = x_try_d[i] - x_d[i];
                Yslot[i] = g_try_d[i] - g_d[i];
            }
            if (lane == 0) rho_d[slot] = 1.0 / sy;
            ++k;
        } else {
            k = 0;
        }
        __syncwarp();

        for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) {
            x_d[i] = x_try_d[i];
            g_d[i] = g_try_d[i];
        }
        f_cur = f_new;
        __syncwarp();

        if (dx < UNMIX_INNER_DX_TOL || df < UNMIX_INNER_DF_TOL) break;
    }
}

__device__ void solve_one_pixel_warp(
    int pixel_idx, int L, int dim,
    double* __restrict__ x_d,
    double* __restrict__ g_d,
    double* __restrict__ S_d,
    double* __restrict__ Y_d,
    double* __restrict__ rho_d,
    double* __restrict__ q_d,
    double* __restrict__ d_d,
    double* __restrict__ x_try_d,
    double* __restrict__ g_try_d,
    double* __restrict__ x_alm_start_d,
    const double* __restrict__ kernel_mu,
    const double* __restrict__ kernel_sigma_inv,
    const double* __restrict__ target_color,
    const double* __restrict__ target_alphas,
    const double* __restrict__ initial_colors,
    int use_sparsity, int use_target_alphas,
    double* __restrict__ out_sol,
    int max_alm_iter, int max_inner_iter,
    double epsilon, double sigma_sparsity,
    double lambda_init, double lo_init,
    int lane)
{
    if (use_target_alphas && initial_colors) {
        for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) {
            x_d[i] = (i < L) ? target_alphas[i] : initial_colors[i - L];
        }
    } else {
        double min_dist = 1e30;
        int closest = 0;
        for (int i = 0; i < L; ++i) {
            const double* mu = kernel_mu + i * 3;
            const double dr = target_color[0] - mu[0];
            const double dg = target_color[1] - mu[1];
            const double db = target_color[2] - mu[2];
            const double* s = kernel_sigma_inv + i * 9;
            const double maha =
                dr * (s[0] * dr + s[1] * dg + s[2] * db) +
                dg * (s[3] * dr + s[4] * dg + s[5] * db) +
                db * (s[6] * dr + s[7] * dg + s[8] * db);
            if (maha < min_dist) { min_dist = maha; closest = i; }
        }
        for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) {
            if (i < L) {
                x_d[i] = (i == closest) ? 1.0 : 0.0;
            } else {
                const int li = (i - L) / 3;
                const int ci = (i - L) % 3;
                double v = (li == closest) ? target_color[ci] : kernel_mu[li * 3 + ci];
                x_d[i] = fmin(fmax(v, 0.0), 1.0);
            }
        }
    }
    __syncwarp();

    double lambda[4] = { lambda_init, lambda_init, lambda_init, lambda_init };
    double lo = lo_init;

    double cons_old[4];
    compute_constraint_warp(x_d, L, target_color, target_alphas, use_target_alphas, cons_old, lane);
    __syncwarp();

    for (int alm = 0; alm < max_alm_iter; ++alm) {
        for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) x_alm_start_d[i] = x_d[i];
        __syncwarp();

        lbfgs_minimize_warp(x_d, g_d, L, dim,
                            kernel_mu, kernel_sigma_inv,
                            target_color, target_alphas,
                            use_sparsity, use_target_alphas,
                            sigma_sparsity, lambda, lo,
                            S_d, Y_d, rho_d,
                            q_d, d_d, x_try_d, g_try_d,
                            max_inner_iter, lane);

        double cons_new[4];
        compute_constraint_warp(x_d, L, target_color, target_alphas, use_target_alphas, cons_new, lane);
        __syncwarp();

        double cons_norm_sq = 0.0, old_norm_sq = 0.0;
        bool lam_finite = true;
        for (int j = 0; j < 4; ++j) {
            lambda[j] += lo * cons_new[j];
            if (!is_finite(lambda[j])) lam_finite = false;
            cons_norm_sq += cons_new[j] * cons_new[j];
            old_norm_sq  += cons_old[j] * cons_old[j];
        }
        if (!lam_finite) {
            for (int j = 0; j < 4; ++j) lambda[j] = lambda_init;
        }
        const double cons_norm = sqrt(cons_norm_sq);
        const double old_norm  = sqrt(old_norm_sq);
        if (!is_finite(cons_norm) || cons_norm > UNMIX_ALM_GAMMA * old_norm) lo *= UNMIX_ALM_BETA;
        if (!is_finite(lo) || lo > UNMIX_LO_MAX) lo = UNMIX_LO_MAX;
        for (int j = 0; j < 4; ++j) cons_old[j] = cons_new[j];

        double dx_outer_sq = 0.0;
        for (int i = lane; i < dim; i += UNMIX_WARP_SIZE) {
            const double dxi = x_d[i] - x_alm_start_d[i];
            dx_outer_sq += dxi * dxi;
        }
        dx_outer_sq = warp_sum_broadcast(dx_outer_sq);

        const bool is_unchanged = (dx_outer_sq < epsilon);
        const bool is_satisfied = (cons_norm < epsilon);
        if (is_unchanged && is_satisfied) break;
    }

    for (int i = lane; i < L; i += UNMIX_WARP_SIZE) {
        double a = fmin(fmax(x_d[i], 0.0), 1.0);
        out_sol[i] = a;
        if (a < 1e-4) {
            out_sol[L + i*3 + 0] = kernel_mu[i*3 + 0];
            out_sol[L + i*3 + 1] = kernel_mu[i*3 + 1];
            out_sol[L + i*3 + 2] = kernel_mu[i*3 + 2];
        } else {
            out_sol[L + i*3 + 0] = fmin(fmax(x_d[L + i*3 + 0], 0.0), 1.0);
            out_sol[L + i*3 + 1] = fmin(fmax(x_d[L + i*3 + 1], 0.0), 1.0);
            out_sol[L + i*3 + 2] = fmin(fmax(x_d[L + i*3 + 2], 0.0), 1.0);
        }
    }
    __syncwarp();
}

__global__ void batch_solve_kernel(
    int P, int L,
    const double* __restrict__ d_kernel_mu,
    const double* __restrict__ d_kernel_sigma_inv,
    const double* __restrict__ d_target_colors,
    const double* __restrict__ d_target_alphas,
    const double* __restrict__ d_initial_colors,
    const int* __restrict__ d_use_sparsity,
    const int* __restrict__ d_use_target_alphas,
    double* __restrict__ d_solution,
    int max_alm_iter, int max_inner_iter,
    double epsilon, double sigma_sparsity,
    double lambda_init, double lo_init)
{
    extern __shared__ char smem_raw[];

    const int lane            = threadIdx.x & (UNMIX_WARP_SIZE - 1);
    const int warp_id_block   = threadIdx.x / UNMIX_WARP_SIZE;
    const int warp_id_grid    = (blockIdx.x * blockDim.x + threadIdx.x) / UNMIX_WARP_SIZE;
    if (warp_id_grid >= P) return;

    const int dim = 4 * L;
    const int per_warp_double = 2 * UNMIX_LBFGS_M * dim + 7 * dim + UNMIX_LBFGS_M;
    const int per_warp_bytes  = ((per_warp_double * (int)sizeof(double) + 7) / 8) * 8;

    char* base = smem_raw + warp_id_block * per_warp_bytes;
    double* x_d         = reinterpret_cast<double*>(base);
    double* g_d         = x_d + dim;
    double* S_d         = g_d + dim;
    double* Y_d         = S_d + UNMIX_LBFGS_M * dim;
    double* rho_d       = Y_d + UNMIX_LBFGS_M * dim;
    double* q_d         = rho_d + UNMIX_LBFGS_M;
    double* d_d         = q_d + dim;
    double* x_try_d     = d_d + dim;
    double* g_try_d     = x_try_d + dim;
    double* x_alm_start_d = g_try_d + dim;

    const double* target_color = d_target_colors + (size_t)warp_id_grid * 3;
    const double* target_alphas = d_target_alphas
        ? d_target_alphas + (size_t)warp_id_grid * L : nullptr;
    const double* initial_colors = d_initial_colors
        ? d_initial_colors + (size_t)warp_id_grid * L * 3 : nullptr;

    solve_one_pixel_warp(
        warp_id_grid, L, dim,
        x_d, g_d, S_d, Y_d, rho_d,
        q_d, d_d, x_try_d, g_try_d, x_alm_start_d,
        d_kernel_mu, d_kernel_sigma_inv,
        target_color, target_alphas, initial_colors,
        d_use_sparsity[warp_id_grid],
        d_use_target_alphas[warp_id_grid],
        d_solution + (size_t)warp_id_grid * dim,
        max_alm_iter, max_inner_iter,
        epsilon, sigma_sparsity,
        lambda_init, lo_init,
        lane);
}

}

BatchSolveOutput solve_batch(const BatchSolveInput& input) {
    const int P = input.num_pixels;
    const int L = input.num_layers;
    if (P <= 0 || L <= 0) return {};

    if (input.target_colors.size() < (size_t)P * 3) {
        std::fprintf(stderr, "[batch_solver] target_colors too small\n");
        return {};
    }
    if ((int)input.use_sparsity.size() < P || (int)input.use_target_alphas.size() < P) {
        std::fprintf(stderr, "[batch_solver] flags too small\n");
        return {};
    }
    if (input.kernel_mu.size() < (size_t)L * 3 || input.kernel_sigma_inv.size() < (size_t)L * 9) {
        std::fprintf(stderr, "[batch_solver] kernels too small\n");
        return {};
    }

    bool any_ta = false;
    for (int p = 0; p < P; ++p) {
        if (input.use_target_alphas[p]) { any_ta = true; break; }
    }
    if (any_ta && input.target_alphas.size() < (size_t)P * L) {
        std::fprintf(stderr, "[batch_solver] target_alphas too small\n");
        return {};
    }
    if (!input.initial_colors.empty() && input.initial_colors.size() < (size_t)P * L * 3) {
        std::fprintf(stderr, "[batch_solver] initial_colors too small\n");
        return {};
    }

    BatchSolveOutput out;
    out.solution.resize((size_t)P * 4 * L);

    const int dim = 4 * L;

    double *d_mu = nullptr, *d_sinv = nullptr;
    double *d_tc = nullptr, *d_ta = nullptr, *d_ic = nullptr;
    double *d_sol = nullptr;
    int    *d_us = nullptr, *d_uta = nullptr;

    const size_t sz_mu   = (size_t)L * 3 * sizeof(double);
    const size_t sz_sinv = (size_t)L * 9 * sizeof(double);
    const size_t sz_tc   = (size_t)P * 3 * sizeof(double);
    const size_t sz_ta   = (size_t)P * L * sizeof(double);
    const size_t sz_ic   = (size_t)P * L * 3 * sizeof(double);
    const size_t sz_sol  = (size_t)P * dim * sizeof(double);
    const size_t sz_flag = (size_t)P * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_mu,   sz_mu));
    CUDA_CHECK(cudaMalloc(&d_sinv, sz_sinv));
    CUDA_CHECK(cudaMalloc(&d_tc,   sz_tc));
    CUDA_CHECK(cudaMalloc(&d_sol,  sz_sol));
    CUDA_CHECK(cudaMalloc(&d_us,   sz_flag));
    CUDA_CHECK(cudaMalloc(&d_uta,  sz_flag));

    CUDA_CHECK(cudaMemcpy(d_mu,   input.kernel_mu.data(),         sz_mu,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sinv, input.kernel_sigma_inv.data(),  sz_sinv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tc,   input.target_colors.data(),     sz_tc,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_us,   input.use_sparsity.data(),      sz_flag, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uta,  input.use_target_alphas.data(), sz_flag, cudaMemcpyHostToDevice));

    if (any_ta) {
        CUDA_CHECK(cudaMalloc(&d_ta, sz_ta));
        CUDA_CHECK(cudaMemcpy(d_ta, input.target_alphas.data(), sz_ta, cudaMemcpyHostToDevice));
    }
    if (!input.initial_colors.empty()) {
        CUDA_CHECK(cudaMalloc(&d_ic, sz_ic));
        CUDA_CHECK(cudaMemcpy(d_ic, input.initial_colors.data(), sz_ic, cudaMemcpyHostToDevice));
    }

    const int per_warp_double = 2 * UNMIX_LBFGS_M * dim + 7 * dim + UNMIX_LBFGS_M;
    const int per_warp_bytes  = ((per_warp_double * (int)sizeof(double) + 7) / 8) * 8;

    int warps_per_block = 12;
    while (warps_per_block > 1 && (size_t)per_warp_bytes * warps_per_block > 48000) {
        warps_per_block /= 2;
    }
    const size_t smem_bytes = (size_t)warps_per_block * per_warp_bytes;
    if (smem_bytes > 48 * 1024) {
        std::fprintf(stderr, "[batch_solver] smem %zu exceeds 48 KB (L=%d)\n", smem_bytes, L);
        cudaFree(d_mu); cudaFree(d_sinv); cudaFree(d_tc);
        if (d_ta) cudaFree(d_ta);
        if (d_ic) cudaFree(d_ic);
        cudaFree(d_sol); cudaFree(d_us); cudaFree(d_uta);
        return {};
    }

    const int blocks  = (P + warps_per_block - 1) / warps_per_block;
    const int threads = warps_per_block * UNMIX_WARP_SIZE;

    batch_solve_kernel<<<blocks, threads, smem_bytes>>>(
        P, L,
        d_mu, d_sinv, d_tc, d_ta, d_ic,
        d_us, d_uta, d_sol,
        input.max_alm_iter, input.max_inner_iter,
        input.epsilon, input.sigma_sparsity,
        input.lambda_init, input.lo_init);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out.solution.data(), d_sol, sz_sol, cudaMemcpyDeviceToHost));

    cudaFree(d_mu);
    cudaFree(d_sinv);
    cudaFree(d_tc);
    if (d_ta) cudaFree(d_ta);
    if (d_ic) cudaFree(d_ic);
    cudaFree(d_sol);
    cudaFree(d_us);
    cudaFree(d_uta);

    return out;
}

}
}