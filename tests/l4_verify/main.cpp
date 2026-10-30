#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <system_error>
#include "pipeline.hpp"
#include "io.hpp"
#include "backend_cpu.hpp"
#include "backend_cuda.hpp"
#include "cuda_backend.hpp"

using namespace unmixing;

struct ChannelStats {
    double mae = 0.0;
    double max_diff = 0.0;
    size_t count = 0;
};

static void compare_buffer(const ImageBuffer& a, const ImageBuffer& b, ChannelStats stats[4]) {
    if (a.width != b.width || a.height != b.height) {
        std::fprintf(stderr, "L4 dimension mismatch: %dx%d vs %dx%d\n",
                     a.width, a.height, b.width, b.height);
        return;
    }
    const std::uint8_t* pa = static_cast<const std::uint8_t*>(a.bits);
    const std::uint8_t* pb = static_cast<const std::uint8_t*>(b.bits);
    const size_t n = (size_t)a.width * a.height;
    for (size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 4; ++c) {
            const double d = std::fabs((double)pa[i * 4 + c] - (double)pb[i * 4 + c]);
            stats[c].mae += d;
            stats[c].max_diff = std::max(stats[c].max_diff, d);
            stats[c].count++;
        }
    }
}

static bool check_stats(const char* label, ChannelStats stats[4]) {
    const char* names[4] = {"B", "G", "R", "A"};
    bool pass = true;
    for (int c = 0; c < 4; ++c) {
        const double mae = stats[c].count ? stats[c].mae / stats[c].count : 0.0;
        std::printf("  %s %s: MAE=%.3f Max=%.0f\n", label, names[c], mae, stats[c].max_diff);
        if (mae >= 0.5 || stats[c].max_diff >= 2.0) pass = false;
    }
    return pass;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: l4_verify <input.bmp>\n");
        return 1;
    }

    int w = 0, h = 0;
    std::vector<std::uint8_t> bgra = io::load_bgra_from_bmp(argv[1], w, h);
    if (bgra.empty()) {
        std::fprintf(stderr, "L4 FAIL: cannot load input\n");
        return 1;
    }

    UnmixedData cpu_out;
    run(bgra.data(), w, h, cpu_out, cpu_backend(), false);

    UnmixedData cuda_out;
    run(bgra.data(), w, h, cuda_out, cuda_backend(), false);

    if (cpu_out.data_num != cuda_out.data_num) {
        std::fprintf(stderr, "L4 FAIL: layer count %d vs %d\n", cpu_out.data_num, cuda_out.data_num);
        free_unmixed_data(cpu_out);
        free_unmixed_data(cuda_out);
        return 1;
    }

    bool all_pass = true;
    for (int i = 0; i < cpu_out.data_num; ++i) {
        ChannelStats s_output[4]{}, s_alpha[4]{}, s_overlay[4]{}, s_refined[4]{};
        compare_buffer(cpu_out.data[i].output,  cuda_out.data[i].output,  s_output);
        compare_buffer(cpu_out.data[i].output_a, cuda_out.data[i].output_a, s_alpha);
        compare_buffer(cpu_out.data[i].overlay, cuda_out.data[i].overlay, s_overlay);
        compare_buffer(cpu_out.data[i].refined, cuda_out.data[i].refined, s_refined);
        std::printf("Layer %d\n", i);
        all_pass &= check_stats("output",  s_output);
        all_pass &= check_stats("alpha",   s_alpha);
        all_pass &= check_stats("overlay", s_overlay);
        all_pass &= check_stats("refined", s_refined);
    }

    free_unmixed_data(cpu_out);
    free_unmixed_data(cuda_out);

    std::printf(all_pass ? "L4 PASS\n" : "L4 FAIL\n");
    return all_pass ? 0 : 1;
}