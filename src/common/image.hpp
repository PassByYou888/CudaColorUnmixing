#pragma once
#include <vector>
#include <cstddef>
#include <cassert>

namespace unmixing {

class GrayImage {
public:
    GrayImage() = default;
    GrayImage(int w, int h, double v = 0.0)
        : w_(w), h_(h), data_(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), v) {}

    int width()  const { return w_; }
    int height() const { return h_; }

    double*       data()       { return data_.data(); }
    const double* data() const { return data_.data(); }

    double& at(int x, int y) {
        assert(x >= 0 && x < w_ && y >= 0 && y < h_);
        return data_[static_cast<std::size_t>(y) * w_ + x];
    }
    double at(int x, int y) const {
        assert(x >= 0 && x < w_ && y >= 0 && y < h_);
        return data_[static_cast<std::size_t>(y) * w_ + x];
    }

    double*       row(int y)       { return data_.data() + static_cast<std::size_t>(y) * w_; }
    const double* row(int y) const { return data_.data() + static_cast<std::size_t>(y) * w_; }

private:
    int w_ = 0;
    int h_ = 0;
    std::vector<double> data_;
};

class ColorImage {
public:
    ColorImage() = default;
    ColorImage(int w, int h) : w_(w), h_(h), r_(w, h), g_(w, h), b_(w, h), a_(w, h) {}

    int width()  const { return w_; }
    int height() const { return h_; }

    GrayImage&       r()       { return r_; }
    GrayImage&       g()       { return g_; }
    GrayImage&       b()       { return b_; }
    GrayImage&       a()       { return a_; }
    const GrayImage& r() const { return r_; }
    const GrayImage& g() const { return g_; }
    const GrayImage& b() const { return b_; }
    const GrayImage& a() const { return a_; }

private:
    int w_ = 0;
    int h_ = 0;
    GrayImage r_, g_, b_, a_;
};

}