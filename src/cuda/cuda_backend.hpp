#pragma once
#include <vector>
#include <Eigen/Core>
#include "image.hpp"

namespace unmixing {
namespace cuda {

bool available();

GrayImage luminance(const ColorImage& img);

GrayImage sobel_x(const GrayImage& img);
GrayImage sobel_y(const GrayImage& img);
GrayImage gradient_magnitude(const GrayImage& img);

GrayImage box_filter(const GrayImage& img, int radius);

GrayImage guided_filter_kernel(const GrayImage& img,
                               int cx, int cy, int radius,
                               double epsilon, bool force_positive);

void compute_normal_distribution(const ColorImage& img, const GrayImage& weight,
                                 Eigen::Vector3d& mu, Eigen::Matrix3d& sigma);

GrayImage guided_filter(const GrayImage& input,
                        const GrayImage& guide_r,
                        const GrayImage& guide_g,
                        const GrayImage& guide_b,
                        int radius, double epsilon);

struct GfGuidanceStats {
    double* d_mean_r = nullptr;
    double* d_mean_g = nullptr;
    double* d_mean_b = nullptr;
    double* d_var_rr = nullptr;
    double* d_var_rg = nullptr;
    double* d_var_rb = nullptr;
    double* d_var_gg = nullptr;
    double* d_var_gb = nullptr;
    double* d_var_bb = nullptr;
    int W = 0, H = 0;

    bool valid() const { return d_mean_r != nullptr; }
};

void compute_gf_guidance_stats(const ColorImage& guide,
                               int radius, int W, int H,
                               GfGuidanceStats& stats);

void free_gf_guidance_stats(GfGuidanceStats& stats);

GrayImage guided_filter_with_stats(const GrayImage& input,
                                    const ColorImage& guide,
                                    const GfGuidanceStats& stats,
                                    int radius, double epsilon);

}
}