#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include "cuda_backend.hpp"
#include "types.hpp"

namespace unmixing {
namespace cuda {

namespace {

#define CUDA_OK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "[cuda] %s failed: %s (line %d)\n", \
                     #call, cudaGetErrorString(_e), __LINE__); \
        std::abort(); \
    } \
} while (0)

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ void inv3x3_sym(
    double a00, double a01, double a02,
    double a11, double a12, double a22,
    double* out)
{
    const double c00 = a11 * a22 - a12 * a12;
    const double c01 = a02 * a12 - a01 * a22;
    const double c02 = a01 * a12 - a02 * a11;
    const double c11 = a00 * a22 - a02 * a02;
    const double c12 = a02 * a01 - a00 * a12;
    const double c22 = a00 * a11 - a01 * a01;
    const double det = a00 * c00 + a01 * c01 + a02 * c02;
    const double inv = 1.0 / det;
    out[0] = c00 * inv;
    out[1] = c01 * inv;
    out[2] = c02 * inv;
    out[3] = c01 * inv;
    out[4] = c11 * inv;
    out[5] = c12 * inv;
    out[6] = c02 * inv;
    out[7] = c12 * inv;
    out[8] = c22 * inv;
}

__device__ __forceinline__ double rect_sum(const double* sat, int W,
                                            int x0, int y0, int x1, int y1) {
    double s = sat[y1 * W + x1];
    if (x0 > 0) s -= sat[y1 * W + (x0 - 1)];
    if (y0 > 0) s -= sat[(y0 - 1) * W + x1];
    if (x0 > 0 && y0 > 0) s += sat[(y0 - 1) * W + (x0 - 1)];
    return s;
}

__global__ void k_luminance(const double* __restrict__ r,
                            const double* __restrict__ g,
                            const double* __restrict__ b,
                            double* __restrict__ out,
                            int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 0.2126 * r[i] + 0.7152 * g[i] + 0.0722 * b[i];
}

__global__ void k_sobel_x(const double* __restrict__ in, double* __restrict__ out, int W, int H) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int ym = clampi(y - 1, 0, H - 1);
    const int yp = clampi(y + 1, 0, H - 1);
    const int xm = clampi(x - 1, 0, W - 1);
    const int xp = clampi(x + 1, 0, W - 1);
    const double* r0 = in + (size_t)ym * W;
    const double* r1 = in + (size_t)y  * W;
    const double* r2 = in + (size_t)yp * W;
    out[(size_t)y * W + x] = (r0[xm] - r0[xp]) + 2.0 * (r1[xm] - r1[xp]) + (r2[xm] - r2[xp]);
}

__global__ void k_sobel_y(const double* __restrict__ in, double* __restrict__ out, int W, int H) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int ym = clampi(y - 1, 0, H - 1);
    const int yp = clampi(y + 1, 0, H - 1);
    const int xm = clampi(x - 1, 0, W - 1);
    const int xp = clampi(x + 1, 0, W - 1);
    const double* r0 = in + (size_t)ym * W;
    const double* r2 = in + (size_t)yp * W;
    out[(size_t)y * W + x] = (r0[xm] + 2.0 * r0[x] + r0[xp]) - (r2[xm] + 2.0 * r2[x] + r2[xp]);
}

__global__ void k_gradient_magnitude(const double* __restrict__ gx,
                                     const double* __restrict__ gy,
                                     double* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = std::sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
}

__global__ void k_box_h(const double* __restrict__ in, double* __restrict__ out,
                        int W, int H, int r, double inv) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const double* row = in + (size_t)y * W;
    double s = 0.0;
    for (int k = -r; k <= r; ++k) s += row[clampi(x + k, 0, W - 1)];
    out[(size_t)y * W + x] = s * inv;
}

__global__ void k_box_v(const double* __restrict__ in, double* __restrict__ out,
                        int W, int H, int r, double inv) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    double s = 0.0;
    for (int k = -r; k <= r; ++k) s += in[(size_t)clampi(y + k, 0, H - 1) * W + x];
    out[(size_t)y * W + x] = s * inv;
}

__global__ void k_normal_mu_block(const double* __restrict__ r,
                                  const double* __restrict__ g,
                                  const double* __restrict__ b,
                                  const double* __restrict__ w,
                                  double* __restrict__ block_sums,
                                  int n)
{
    __shared__ double s[3][256];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    double wr = 0.0, wg = 0.0, wb = 0.0;
    if (i < n) {
        const double wi = w[i];
        if (wi != 0.0) {
            wr = wi * r[i];
            wg = wi * g[i];
            wb = wi * b[i];
        }
    }
    s[0][tid] = wr;
    s[1][tid] = wg;
    s[2][tid] = wb;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s[0][tid] += s[0][tid + stride];
            s[1][tid] += s[1][tid + stride];
            s[2][tid] += s[2][tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_sums[blockIdx.x * 3 + 0] = s[0][0];
        block_sums[blockIdx.x * 3 + 1] = s[1][0];
        block_sums[blockIdx.x * 3 + 2] = s[2][0];
    }
}

__global__ void k_normal_sigma_block(const double* __restrict__ r,
                                     const double* __restrict__ g,
                                     const double* __restrict__ b,
                                     const double* __restrict__ w,
                                     const double* __restrict__ mu,
                                     double* __restrict__ block_sums,
                                     int n)
{
    __shared__ double s[6][256];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    double srr = 0.0, sgg = 0.0, sbb = 0.0, srg = 0.0, srb = 0.0, sgb = 0.0;
    if (i < n) {
        const double wi = w[i];
        if (wi != 0.0) {
            const double dr = r[i] - mu[0];
            const double dg = g[i] - mu[1];
            const double db = b[i] - mu[2];
            srr = wi * dr * dr;
            sgg = wi * dg * dg;
            sbb = wi * db * db;
            srg = wi * dr * dg;
            srb = wi * dr * db;
            sgb = wi * dg * db;
        }
    }
    s[0][tid] = srr; s[1][tid] = sgg; s[2][tid] = sbb;
    s[3][tid] = srg; s[4][tid] = srb; s[5][tid] = sgb;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int j = 0; j < 6; ++j) s[j][tid] += s[j][tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int j = 0; j < 6; ++j) block_sums[blockIdx.x * 6 + j] = s[j][0];
    }
}

__global__ void k_elem_mul(const double* __restrict__ a, const double* __restrict__ b,
                           double* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] * b[i];
}

__global__ void k_elem_sub(const double* __restrict__ a, const double* __restrict__ b,
                           double* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] - b[i];
}

__global__ void k_elem_sub_inplace(double* __restrict__ a, const double* __restrict__ b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = a[i] - b[i];
}

__global__ void k_gf_compute_a(
    const double* __restrict__ cov_rp, const double* __restrict__ cov_gp,
    const double* __restrict__ cov_bp,
    const double* __restrict__ var_rr, const double* __restrict__ var_rg,
    const double* __restrict__ var_rb,
    const double* __restrict__ var_gg, const double* __restrict__ var_gb,
    const double* __restrict__ var_bb,
    double* __restrict__ a_r, double* __restrict__ a_g, double* __restrict__ a_b,
    int n, double eps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double inv[9];
    inv3x3_sym(var_rr[i] + eps, var_rg[i], var_rb[i],
               var_gg[i] + eps, var_gb[i], var_bb[i] + eps, inv);
    const double c0 = cov_rp[i], c1 = cov_gp[i], c2 = cov_bp[i];
    a_r[i] = inv[0] * c0 + inv[1] * c1 + inv[2] * c2;
    a_g[i] = inv[3] * c0 + inv[4] * c1 + inv[5] * c2;
    a_b[i] = inv[6] * c0 + inv[7] * c1 + inv[8] * c2;
}

__global__ void k_gf_compute_b(
    const double* __restrict__ mean_p,
    const double* __restrict__ a_r, const double* __restrict__ a_g,
    const double* __restrict__ a_b,
    const double* __restrict__ mean_r, const double* __restrict__ mean_g,
    const double* __restrict__ mean_b,
    double* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = mean_p[i] - a_r[i] * mean_r[i] - a_g[i] * mean_g[i] - a_b[i] * mean_b[i];
}

__global__ void k_gf_compute_q(
    const double* __restrict__ q_base,
    const double* __restrict__ a_r, const double* __restrict__ a_g,
    const double* __restrict__ a_b,
    const double* __restrict__ guide_r, const double* __restrict__ guide_g,
    const double* __restrict__ guide_b,
    double* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = q_base[i] + a_r[i] * guide_r[i] + a_g[i] * guide_g[i] + a_b[i] * guide_b[i];
}

__global__ void k_compute_AC_sat(
    const double* __restrict__ mean_I, const double* __restrict__ var_I,
    double* __restrict__ A, double* __restrict__ C,
    int n, double I_seed, double eps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double mu = mean_I[i];
    const double v  = var_I[i];
    const double d  = (I_seed - mu) / (eps + v);
    A[i] = d;
    C[i] = d * mu;
}

__global__ void k_sat_rowscan(const double* __restrict__ src, double* __restrict__ dst,
                              int W, int H) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= H) return;
    const double* s = src + (size_t)y * W;
    double* d = dst + (size_t)y * W;
    double acc = 0.0;
    for (int x = 0; x < W; ++x) {
        acc += s[x];
        d[x] = acc;
    }
}

__global__ void k_sat_colscan_inplace(double* __restrict__ data, int W, int H) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= W) return;
    double acc = 0.0;
    for (int y = 0; y < H; ++y) {
        const size_t idx = (size_t)y * W + x;
        acc += data[idx];
        data[idx] = acc;
    }
}

__global__ void k_gfk_sat_query(
    const double* __restrict__ img,
    const double* __restrict__ sat_A,
    const double* __restrict__ sat_C,
    double* __restrict__ weight,
    int W, int H, int cx, int cy, int r, int force_positive)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    if (x < cx - r || x > cx + r || y < cy - r || y > cy + r) return;

    const int kx0 = max(0, max(x - r, cx - r));
    const int kx1 = min(W - 1, min(x + r, cx + r));
    const int ky0 = max(0, max(y - r, cy - r));
    const int ky1 = min(H - 1, min(y + r, cy + r));
    if (kx0 > kx1 || ky0 > ky1) { weight[(size_t)y * W + x] = 0.0; return; }

    const double SA = rect_sum(sat_A, W, kx0, ky0, kx1, ky1);
    const double SC = rect_sum(sat_C, W, kx0, ky0, kx1, ky1);
    const int N = (kx1 - kx0 + 1) * (ky1 - ky0 + 1);
    const double I_j = img[(size_t)y * W + x];
    double w = (double)N + I_j * SA - SC;
    if (force_positive && w < 0.0) w = 0.0;
    weight[(size_t)y * W + x] = w;
}

dim3 block_2d() { return dim3(16, 16, 1); }
dim3 grid_2d(int W, int H) { return dim3((W + 15) / 16, (H + 15) / 16, 1); }
dim3 block_1d(int n) { return dim3(std::min(256, n), 1, 1); }
dim3 grid_1d(int n) { return dim3((n + 255) / 256, 1, 1); }

void launch_box_filter_device(const double* d_in, double* d_out, double* d_tmp,
                              int W, int H, int r) {
    const double inv = 1.0 / (2 * r + 1);
    k_box_h<<<grid_2d(W, H), block_2d()>>>(d_in, d_tmp, W, H, r, inv);
    CUDA_OK(cudaGetLastError());
    k_box_v<<<grid_2d(W, H), block_2d()>>>(d_tmp, d_out, W, H, r, inv);
    CUDA_OK(cudaGetLastError());
}

double* upload(const GrayImage& img) {
    const size_t bytes = (size_t)img.width() * img.height() * sizeof(double);
    double* d = nullptr;
    CUDA_OK(cudaMalloc(&d, bytes));
    CUDA_OK(cudaMemcpy(d, img.data(), bytes, cudaMemcpyHostToDevice));
    return d;
}

double* device_alloc(int W, int H) {
    double* d = nullptr;
    CUDA_OK(cudaMalloc(&d, (size_t)W * H * sizeof(double)));
    return d;
}

void download(double* d, GrayImage& img) {
    const size_t bytes = (size_t)img.width() * img.height() * sizeof(double);
    CUDA_OK(cudaMemcpy(img.data(), d, bytes, cudaMemcpyDeviceToHost));
}

void safe_free(double* d) { if (d) cudaFree(d); }

void build_sat_device(const double* d_src, double* d_sat, double* d_row,
                      int W, int H) {
    k_sat_rowscan<<<grid_1d(H), block_1d(H)>>>(d_src, d_row, W, H);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(d_sat, d_row, (size_t)W * H * sizeof(double), cudaMemcpyDeviceToDevice));
    k_sat_colscan_inplace<<<grid_1d(W), block_1d(W)>>>(d_sat, W, H);
    CUDA_OK(cudaGetLastError());
}

}

bool available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
    return (prop.major * 10 + prop.minor) >= 52;
}

GrayImage luminance(const ColorImage& img) {
    const int W = img.width(), H = img.height();
    const int n = W * H;
    GrayImage out(W, H);
    const size_t bytes = (size_t)n * sizeof(double);
    double* d_r = nullptr; double* d_g = nullptr; double* d_b = nullptr; double* d_o = nullptr;
    CUDA_OK(cudaMalloc(&d_r, bytes));
    CUDA_OK(cudaMalloc(&d_g, bytes));
    CUDA_OK(cudaMalloc(&d_b, bytes));
    CUDA_OK(cudaMalloc(&d_o, bytes));
    CUDA_OK(cudaMemcpy(d_r, img.r().data(), bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_g, img.g().data(), bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_b, img.b().data(), bytes, cudaMemcpyHostToDevice));
    k_luminance<<<grid_1d(n), block_1d(n)>>>(d_r, d_g, d_b, d_o, n);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(out.data(), d_o, bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_r); cudaFree(d_g); cudaFree(d_b); cudaFree(d_o);
    return out;
}

GrayImage sobel_x(const GrayImage& img) {
    const int W = img.width(), H = img.height();
    GrayImage out(W, H);
    double* d_in = upload(img);
    double* d_out = device_alloc(W, H);
    k_sobel_x<<<grid_2d(W, H), block_2d()>>>(d_in, d_out, W, H);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    download(d_out, out);
    safe_free(d_in); safe_free(d_out);
    return out;
}

GrayImage sobel_y(const GrayImage& img) {
    const int W = img.width(), H = img.height();
    GrayImage out(W, H);
    double* d_in = upload(img);
    double* d_out = device_alloc(W, H);
    k_sobel_y<<<grid_2d(W, H), block_2d()>>>(d_in, d_out, W, H);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    download(d_out, out);
    safe_free(d_in); safe_free(d_out);
    return out;
}

GrayImage gradient_magnitude(const GrayImage& img) {
    const GrayImage gx = sobel_x(img);
    const GrayImage gy = sobel_y(img);
    const int W = img.width(), H = img.height();
    const int n = W * H;
    GrayImage out(W, H);
    const size_t bytes = (size_t)n * sizeof(double);
    double* d_gx = nullptr; double* d_gy = nullptr; double* d_o = nullptr;
    CUDA_OK(cudaMalloc(&d_gx, bytes));
    CUDA_OK(cudaMalloc(&d_gy, bytes));
    CUDA_OK(cudaMalloc(&d_o, bytes));
    CUDA_OK(cudaMemcpy(d_gx, gx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_gy, gy.data(), bytes, cudaMemcpyHostToDevice));
    k_gradient_magnitude<<<grid_1d(n), block_1d(n)>>>(d_gx, d_gy, d_o, n);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(out.data(), d_o, bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_gx); cudaFree(d_gy); cudaFree(d_o);
    return out;
}

GrayImage box_filter(const GrayImage& img, int radius) {
    if (radius <= 0) return img;
    const int W = img.width(), H = img.height();
    GrayImage out(W, H);
    double* d_in = upload(img);
    double* d_tmp = device_alloc(W, H);
    double* d_out = device_alloc(W, H);
    launch_box_filter_device(d_in, d_out, d_tmp, W, H, radius);
    CUDA_OK(cudaDeviceSynchronize());
    download(d_out, out);
    safe_free(d_in); safe_free(d_tmp); safe_free(d_out);
    return out;
}

GrayImage guided_filter_kernel(const GrayImage& img,
                               int cx, int cy, int radius,
                               double epsilon, bool force_positive) {
    const int W = img.width(), H = img.height();
    const int n = W * H;

    double* d_img  = upload(img);
    double* d_sq   = device_alloc(W, H);
    double* d_mean = device_alloc(W, H);
    double* d_corr = device_alloc(W, H);
    double* d_var  = device_alloc(W, H);
    double* d_tmp  = device_alloc(W, H);
    double* d_btmp = device_alloc(W, H);
    double* d_A    = device_alloc(W, H);
    double* d_C    = device_alloc(W, H);
    double* d_satA = device_alloc(W, H);
    double* d_satC = device_alloc(W, H);
    double* d_w    = device_alloc(W, H);

    CUDA_OK(cudaMemset(d_w, 0, (size_t)n * sizeof(double)));

    k_elem_mul<<<grid_1d(n), block_1d(n)>>>(d_img, d_img, d_sq, n);
    CUDA_OK(cudaGetLastError());
    launch_box_filter_device(d_img, d_mean, d_tmp, W, H, radius);
    launch_box_filter_device(d_sq,  d_corr, d_tmp, W, H, radius);

    k_elem_mul<<<grid_1d(n), block_1d(n)>>>(d_mean, d_mean, d_btmp, n);
    CUDA_OK(cudaGetLastError());
    k_elem_sub<<<grid_1d(n), block_1d(n)>>>(d_corr, d_btmp, d_var, n);
    CUDA_OK(cudaGetLastError());

    const double I_seed = img.at(cx, cy);
    k_compute_AC_sat<<<grid_1d(n), block_1d(n)>>>(d_mean, d_var, d_A, d_C, n, I_seed, epsilon);
    CUDA_OK(cudaGetLastError());

    build_sat_device(d_A, d_satA, d_tmp, W, H);
    build_sat_device(d_C, d_satC, d_tmp, W, H);

    k_gfk_sat_query<<<grid_2d(W, H), block_2d()>>>(
        d_img, d_satA, d_satC, d_w, W, H, cx, cy, radius, force_positive ? 1 : 0);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    GrayImage weight(W, H);
    download(d_w, weight);
    safe_free(d_img); safe_free(d_sq); safe_free(d_mean); safe_free(d_corr);
    safe_free(d_var); safe_free(d_tmp); safe_free(d_btmp);
    safe_free(d_A); safe_free(d_C); safe_free(d_satA); safe_free(d_satC); safe_free(d_w);

    double s = 0.0;
    for (int i = 0; i < n; ++i) s += weight.data()[i];
    if (s > 1e-16) {
        const double inv = 1.0 / s;
        for (int i = 0; i < n; ++i) weight.data()[i] *= inv;
    }
    return weight;
}

void compute_normal_distribution(const ColorImage& img, const GrayImage& weight,
                                 Eigen::Vector3d& mu, Eigen::Matrix3d& sigma) {
    const int W = img.width(), H = img.height();
    const int n = W * H;
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    double* d_r = upload(img.r());
    double* d_g = upload(img.g());
    double* d_b = upload(img.b());
    double* d_w = upload(weight);

    double* d_mu_blocks = nullptr;
    double* d_sigma_blocks = nullptr;
    CUDA_OK(cudaMalloc(&d_mu_blocks, (size_t)num_blocks * 3 * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_sigma_blocks, (size_t)num_blocks * 6 * sizeof(double)));

    k_normal_mu_block<<<num_blocks, block_size>>>(d_r, d_g, d_b, d_w, d_mu_blocks, n);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<double> h_mu_blocks((size_t)num_blocks * 3);
    CUDA_OK(cudaMemcpy(h_mu_blocks.data(), d_mu_blocks,
                       (size_t)num_blocks * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    double h_mu[3] = {0.0, 0.0, 0.0};
    for (int b = 0; b < num_blocks; ++b) {
        h_mu[0] += h_mu_blocks[b * 3 + 0];
        h_mu[1] += h_mu_blocks[b * 3 + 1];
        h_mu[2] += h_mu_blocks[b * 3 + 2];
    }

    double* d_mu = nullptr;
    CUDA_OK(cudaMalloc(&d_mu, 3 * sizeof(double)));
    CUDA_OK(cudaMemcpy(d_mu, h_mu, 3 * sizeof(double), cudaMemcpyHostToDevice));

    k_normal_sigma_block<<<num_blocks, block_size>>>(d_r, d_g, d_b, d_w, d_mu, d_sigma_blocks, n);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<double> h_sigma_blocks((size_t)num_blocks * 6);
    CUDA_OK(cudaMemcpy(h_sigma_blocks.data(), d_sigma_blocks,
                       (size_t)num_blocks * 6 * sizeof(double), cudaMemcpyDeviceToHost));

    double h_sigma[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int b = 0; b < num_blocks; ++b) {
        for (int j = 0; j < 6; ++j) h_sigma[j] += h_sigma_blocks[b * 6 + j];
    }

    mu = Eigen::Vector3d(h_mu[0], h_mu[1], h_mu[2]);
    sigma(0,0) = h_sigma[0];
    sigma(0,1) = h_sigma[3];
    sigma(0,2) = h_sigma[4];
    sigma(1,0) = h_sigma[3];
    sigma(1,1) = h_sigma[1];
    sigma(1,2) = h_sigma[5];
    sigma(2,0) = h_sigma[4];
    sigma(2,1) = h_sigma[5];
    sigma(2,2) = h_sigma[2];
    sigma += params::kNormalDistEpsilon * Eigen::Matrix3d::Identity();

    safe_free(d_r); safe_free(d_g); safe_free(d_b); safe_free(d_w);
    cudaFree(d_mu_blocks); cudaFree(d_sigma_blocks);
    cudaFree(d_mu);
}

void compute_gf_guidance_stats(const ColorImage& guide,
                               int radius, int W, int H,
                               GfGuidanceStats& stats) {
    free_gf_guidance_stats(stats);
    const size_t n = (size_t)W * H;

    stats.d_mean_r = device_alloc(W, H);
    stats.d_mean_g = device_alloc(W, H);
    stats.d_mean_b = device_alloc(W, H);
    stats.d_var_rr = device_alloc(W, H);
    stats.d_var_rg = device_alloc(W, H);
    stats.d_var_rb = device_alloc(W, H);
    stats.d_var_gg = device_alloc(W, H);
    stats.d_var_gb = device_alloc(W, H);
    stats.d_var_bb = device_alloc(W, H);
    stats.W = W; stats.H = H;

    double* d_gr = upload(guide.r());
    double* d_gg = upload(guide.g());
    double* d_gb = upload(guide.b());
    double* d_tmp = device_alloc(W, H);
    double* d_btmp = device_alloc(W, H);
    double* d_sq = device_alloc(W, H);

    launch_box_filter_device(d_gr, stats.d_mean_r, d_tmp, W, H, radius);
    launch_box_filter_device(d_gg, stats.d_mean_g, d_tmp, W, H, radius);
    launch_box_filter_device(d_gb, stats.d_mean_b, d_tmp, W, H, radius);

    k_elem_mul<<<grid_1d((int)n), block_1d((int)n)>>>(d_gr, d_gr, d_sq, (int)n);
    CUDA_OK(cudaGetLastError());
    launch_box_filter_device(d_sq, stats.d_var_rr, d_tmp, W, H, radius);
    k_elem_mul<<<grid_1d((int)n), block_1d((int)n)>>>(stats.d_mean_r, stats.d_mean_r, d_btmp, (int)n);
    CUDA_OK(cudaGetLastError());
    k_elem_sub_inplace<<<grid_1d((int)n), block_1d((int)n)>>>(stats.d_var_rr, d_btmp, (int)n);
    CUDA_OK(cudaGetLastError());

    k_elem_mul<<<grid_1d((int)n), block_1d((int)n)>>>(d_gr, d_gg, d_sq, (int)n);
    CUDA_OK(cudaGetLastError());
    launch_box_filter_device(d_sq, stats.d_var_rg, d_tmp, W, H, radius);
    k_elem_mul<<<grid_1d((int)n), block_1d((int)n)>>>(stats.d_mean_r, stats.d_mean_g, d_btmp, (int)n);
    CUDA_OK(cudaGetLastError());
    k_elem_sub_inplace<<<grid_1d((int)n), block_1d((int)n)>>>(stats.d_var_rg, d_btmp, (int)n);
    CUDA_OK(cudaGetLastError());

    k_elem_mul<<<grid_1d((int)n), block_1d((int)n)>>>(d_gr, d_gb, d_sq, (int)n);
    CUDA_OK(cudaGetLastError());
    launch_box_filter_device(d_sq, stats.d_var_rb, d_tmp, W, H, radius);
    k_elem_mul<<<grid_1d((int)n), block_1d((int)n)>>>(stats.d_mean_r, stats.d_mean_b, d_btmp, (int)n);
    CUDA_OK(cudaGetLastError());
    k_elem_sub_inplace<<<grid_1d((int)n), block_1d((int)n)>>>(stats.d_var_rb, d_btmp, (int)n);
    CUDA_OK(cudaGetLastError());

    k_elem_mul<<<grid_1d((int)n), block_1d((int)n)>>>(d_gg, d_gg, d_sq, (int)n);
    CUDA_OK(cudaGetLastError());
    launch_box_filter_device(d_sq, stats.d_var_gg, d_tmp, W, H, radius);
    k_elem_mul<<<grid_1d((int)n), block_1d((int)n)>>>(stats.d_mean_g, stats.d_mean_g, d_btmp, (int)n);
    CUDA_OK(cudaGetLastError());
    k_elem_sub_inplace<<<grid_1d((int)n), block_1d((int)n)>>>(stats.d_var_gg, d_btmp, (int)n);
    CUDA_OK(cudaGetLastError());

    k_elem_mul<<<grid_1d((int)n), block_1d((int)n)>>>(d_gg, d_gb, d_sq, (int)n);
    CUDA_OK(cudaGetLastError());
    launch_box_filter_device(d_sq, stats.d_var_gb, d_tmp, W, H, radius);
    k_elem_mul<<<grid_1d((int)n), block_1d((int)n)>>>(stats.d_mean_g, stats.d_mean_b, d_btmp, (int)n);
    CUDA_OK(cudaGetLastError());
    k_elem_sub_inplace<<<grid_1d((int)n), block_1d((int)n)>>>(stats.d_var_gb, d_btmp, (int)n);
    CUDA_OK(cudaGetLastError());

    k_elem_mul<<<grid_1d((int)n), block_1d((int)n)>>>(d_gb, d_gb, d_sq, (int)n);
    CUDA_OK(cudaGetLastError());
    launch_box_filter_device(d_sq, stats.d_var_bb, d_tmp, W, H, radius);
    k_elem_mul<<<grid_1d((int)n), block_1d((int)n)>>>(stats.d_mean_b, stats.d_mean_b, d_btmp, (int)n);
    CUDA_OK(cudaGetLastError());
    k_elem_sub_inplace<<<grid_1d((int)n), block_1d((int)n)>>>(stats.d_var_bb, d_btmp, (int)n);
    CUDA_OK(cudaGetLastError());

    safe_free(d_gr); safe_free(d_gg); safe_free(d_gb);
    safe_free(d_tmp); safe_free(d_btmp); safe_free(d_sq);
}

void free_gf_guidance_stats(GfGuidanceStats& stats) {
    safe_free(stats.d_mean_r); stats.d_mean_r = nullptr;
    safe_free(stats.d_mean_g); stats.d_mean_g = nullptr;
    safe_free(stats.d_mean_b); stats.d_mean_b = nullptr;
    safe_free(stats.d_var_rr); stats.d_var_rr = nullptr;
    safe_free(stats.d_var_rg); stats.d_var_rg = nullptr;
    safe_free(stats.d_var_rb); stats.d_var_rb = nullptr;
    safe_free(stats.d_var_gg); stats.d_var_gg = nullptr;
    safe_free(stats.d_var_gb); stats.d_var_gb = nullptr;
    safe_free(stats.d_var_bb); stats.d_var_bb = nullptr;
    stats.W = 0; stats.H = 0;
}

GrayImage guided_filter_with_stats(const GrayImage& input,
                                    const ColorImage& guide,
                                    const GfGuidanceStats& stats,
                                    int radius, double epsilon) {
    const int W = input.width(), H = input.height();
    const int n = W * H;
    if (!stats.valid()) {
        std::fprintf(stderr, "[cuda] guided_filter_with_stats: invalid stats\n");
        return GrayImage(W, H);
    }

    double* d_in = upload(input);
    double* d_gr = upload(guide.r());
    double* d_gg = upload(guide.g());
    double* d_gb = upload(guide.b());

    double* d_tmp    = device_alloc(W, H);
    double* d_box    = device_alloc(W, H);
    double* d_mean_p = device_alloc(W, H);
    double* d_mean_rp = device_alloc(W, H);
    double* d_mean_gp = device_alloc(W, H);
    double* d_mean_bp = device_alloc(W, H);
    double* d_cov_rp = device_alloc(W, H);
    double* d_cov_gp = device_alloc(W, H);
    double* d_cov_bp = device_alloc(W, H);
    double* d_a_r = device_alloc(W, H);
    double* d_a_g = device_alloc(W, H);
    double* d_a_b = device_alloc(W, H);
    double* d_b   = device_alloc(W, H);
    double* d_qbase = device_alloc(W, H);
    double* d_abar_r = device_alloc(W, H);
    double* d_abar_g = device_alloc(W, H);
    double* d_abar_b = device_alloc(W, H);
    double* d_q = device_alloc(W, H);

    launch_box_filter_device(d_in, d_mean_p, d_tmp, W, H, radius);

    k_elem_mul<<<grid_1d(n), block_1d(n)>>>(d_gr, d_in, d_box, n);
    CUDA_OK(cudaGetLastError());
    launch_box_filter_device(d_box, d_mean_rp, d_tmp, W, H, radius);
    k_elem_mul<<<grid_1d(n), block_1d(n)>>>(stats.d_mean_r, d_mean_p, d_box, n);
    CUDA_OK(cudaGetLastError());
    k_elem_sub<<<grid_1d(n), block_1d(n)>>>(d_mean_rp, d_box, d_cov_rp, n);
    CUDA_OK(cudaGetLastError());

    k_elem_mul<<<grid_1d(n), block_1d(n)>>>(d_gg, d_in, d_box, n);
    CUDA_OK(cudaGetLastError());
    launch_box_filter_device(d_box, d_mean_gp, d_tmp, W, H, radius);
    k_elem_mul<<<grid_1d(n), block_1d(n)>>>(stats.d_mean_g, d_mean_p, d_box, n);
    CUDA_OK(cudaGetLastError());
    k_elem_sub<<<grid_1d(n), block_1d(n)>>>(d_mean_gp, d_box, d_cov_gp, n);
    CUDA_OK(cudaGetLastError());

    k_elem_mul<<<grid_1d(n), block_1d(n)>>>(d_gb, d_in, d_box, n);
    CUDA_OK(cudaGetLastError());
    launch_box_filter_device(d_box, d_mean_bp, d_tmp, W, H, radius);
    k_elem_mul<<<grid_1d(n), block_1d(n)>>>(stats.d_mean_b, d_mean_p, d_box, n);
    CUDA_OK(cudaGetLastError());
    k_elem_sub<<<grid_1d(n), block_1d(n)>>>(d_mean_bp, d_box, d_cov_bp, n);
    CUDA_OK(cudaGetLastError());

    k_gf_compute_a<<<grid_1d(n), block_1d(n)>>>(
        d_cov_rp, d_cov_gp, d_cov_bp,
        stats.d_var_rr, stats.d_var_rg, stats.d_var_rb,
        stats.d_var_gg, stats.d_var_gb, stats.d_var_bb,
        d_a_r, d_a_g, d_a_b, n, epsilon);
    CUDA_OK(cudaGetLastError());

    k_gf_compute_b<<<grid_1d(n), block_1d(n)>>>(
        d_mean_p, d_a_r, d_a_g, d_a_b,
        stats.d_mean_r, stats.d_mean_g, stats.d_mean_b, d_b, n);
    CUDA_OK(cudaGetLastError());

    launch_box_filter_device(d_b, d_qbase, d_tmp, W, H, radius);
    launch_box_filter_device(d_a_r, d_abar_r, d_tmp, W, H, radius);
    launch_box_filter_device(d_a_g, d_abar_g, d_tmp, W, H, radius);
    launch_box_filter_device(d_a_b, d_abar_b, d_tmp, W, H, radius);

    k_gf_compute_q<<<grid_1d(n), block_1d(n)>>>(
        d_qbase, d_abar_r, d_abar_g, d_abar_b,
        d_gr, d_gg, d_gb, d_q, n);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    GrayImage out(W, H);
    download(d_q, out);

    safe_free(d_in); safe_free(d_gr); safe_free(d_gg); safe_free(d_gb);
    safe_free(d_tmp); safe_free(d_box);
    safe_free(d_mean_p); safe_free(d_mean_rp); safe_free(d_mean_gp); safe_free(d_mean_bp);
    safe_free(d_cov_rp); safe_free(d_cov_gp); safe_free(d_cov_bp);
    safe_free(d_a_r); safe_free(d_a_g); safe_free(d_a_b); safe_free(d_b);
    safe_free(d_qbase);
    safe_free(d_abar_r); safe_free(d_abar_g); safe_free(d_abar_b);
    safe_free(d_q);
    return out;
}

GrayImage guided_filter(const GrayImage& input,
                        const GrayImage& guide_r,
                        const GrayImage& guide_g,
                        const GrayImage& guide_b,
                        int radius, double epsilon) {
    const int W = input.width(), H = input.height();
    ColorImage guide(W, H);
    std::copy(guide_r.data(), guide_r.data() + (size_t)W * H, guide.r().data());
    std::copy(guide_g.data(), guide_g.data() + (size_t)W * H, guide.g().data());
    std::copy(guide_b.data(), guide_b.data() + (size_t)W * H, guide.b().data());

    GfGuidanceStats stats;
    compute_gf_guidance_stats(guide, radius, W, H, stats);
    GrayImage out = guided_filter_with_stats(input, guide, stats, radius, epsilon);
    free_gf_guidance_stats(stats);
    return out;
}

}
}