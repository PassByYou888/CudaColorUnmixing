#pragma once
#include "image.hpp"
#include "types.hpp"

namespace unmixing {
namespace host {

GrayImage luminance(const ColorImage& img);

GrayImage sobel_x(const GrayImage& img);
GrayImage sobel_y(const GrayImage& img);
GrayImage gradient_magnitude(const GrayImage& img);

GrayImage box_filter(const GrayImage& img, int radius);

GrayImage guided_filter(const GrayImage& input,
                        const GrayImage& guide_r,
                        const GrayImage& guide_g,
                        const GrayImage& guide_b,
                        int radius, double epsilon);

GrayImage guided_filter_kernel(const GrayImage& img,
                               int cx, int cy, int radius,
                               double epsilon, bool force_positive);

void compute_normal_distribution(const ColorImage& img, const GrayImage& weight,
                                 Eigen::Vector3d& mu, Eigen::Matrix3d& sigma);

GrayImage multiply(const GrayImage& a, const GrayImage& b);
GrayImage square(const GrayImage& a);
GrayImage subtract(const GrayImage& a, const GrayImage& b);
GrayImage add(const GrayImage& a, const GrayImage& b);

void force_unity(GrayImage& img);
void scale_to_unit(GrayImage& img);

}
}