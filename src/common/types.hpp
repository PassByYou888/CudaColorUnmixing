#pragma once
#include <cstdint>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace unmixing {

struct ImageBuffer {
    void* bits = nullptr;
    int   width  = 0;
    int   height = 0;
};

struct UnmixedLayer {
    ImageBuffer output;
    ImageBuffer output_a;
    ImageBuffer overlay;
    ImageBuffer refined;
    ImageBuffer refined_a;
    ImageBuffer refined_overlay;
};

struct UnmixedData {
    int            data_num = 0;
    UnmixedLayer*  data     = nullptr;
    int            well_represented_map_num = 0;
    ImageBuffer*   well_represented_map     = nullptr;
    int            temporary_weight_map_num = 0;
    ImageBuffer*   temporary_weight_map     = nullptr;
};

struct ColorKernel {
    Eigen::Vector3d mu        = Eigen::Vector3d::Zero();
    Eigen::Matrix3d sigma_inv = Eigen::Matrix3d::Identity();
    int             seed_x    = -1;
    int             seed_y    = -1;

    double squared_mahalanobis(const Eigen::Vector3d& c) const {
        const Eigen::Vector3d d = c - mu;
        return d.dot(sigma_inv * d);
    }
};

namespace params {
constexpr double kTau                    = 5.0;
constexpr int    kNeighborhoodRadius     = 10;
constexpr int    kNumberOfBins           = 10;
constexpr double kAlmostThreshold        = 0.995;
constexpr double kNormalDistEpsilon      = 1e-3;
constexpr double kGuidedFilterKernelEps  = 0.01;
constexpr double kGuidedFilterEps        = 1e-4;
constexpr double kLambdaInit             = 0.1;
constexpr double kLoInit                 = 0.1;
constexpr double kSparsitySigma          = 10.0;
constexpr int    kMaxInnerEval           = 200;
constexpr int    kMaxALMIter             = 100;
constexpr double kALMGamma               = 0.25;
constexpr double kALMBeta                = 10.0;
constexpr double kALMEpsilon             = 1e-8;
}

}