#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include "common/pipeline.hpp"
#include "common/io.hpp"

int main(int argc, char** argv) {
    const int W = (argc > 1) ? std::atoi(argv[1]) : 64;
    const int H = (argc > 2) ? std::atoi(argv[2]) : 64;
    const std::string out_dir = (argc > 3) ? argv[3] : "out";

    std::vector<std::uint8_t> bgra(static_cast<std::size_t>(W) * H * 4);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        const double u = (W > 1) ? static_cast<double>(x) / (W - 1) : 0.0;
        const double v = (H > 1) ? static_cast<double>(y) / (H - 1) : 0.0;
        const double r = u;
        const double g = v;
        const double b = 0.5 + 0.5 * std::sin(u * 6.28 + v * 3.14);
        std::uint8_t* p = &bgra[(static_cast<std::size_t>(y) * W + x) * 4];
        p[0] = static_cast<std::uint8_t>(std::clamp(b, 0.0, 1.0) * 255.0 + 0.5);
        p[1] = static_cast<std::uint8_t>(std::clamp(g, 0.0, 1.0) * 255.0 + 0.5);
        p[2] = static_cast<std::uint8_t>(std::clamp(r, 0.0, 1.0) * 255.0 + 0.5);
        p[3] = 255;
    }

    unmixing::UnmixedData out;
    unmixing::run(bgra.data(), W, H, out, true);

    char path[512];
    for (int i = 0; i < out.data_num; ++i) {
        std::snprintf(path, sizeof(path), "%s/layer_%02d_color.bmp",    out_dir.c_str(), i);
        unmixing::io::write_buffer_bmp(path, out.data[i].output);
        std::snprintf(path, sizeof(path), "%s/layer_%02d_alpha.bmp",    out_dir.c_str(), i);
        unmixing::io::write_buffer_bmp(path, out.data[i].output_a);
        std::snprintf(path, sizeof(path), "%s/layer_%02d_overlay.bmp",  out_dir.c_str(), i);
        unmixing::io::write_buffer_bmp(path, out.data[i].overlay);
        std::snprintf(path, sizeof(path), "%s/refined_%02d_color.bmp",  out_dir.c_str(), i);
        unmixing::io::write_buffer_bmp(path, out.data[i].refined);
        std::snprintf(path, sizeof(path), "%s/refined_%02d_alpha.bmp",  out_dir.c_str(), i);
        unmixing::io::write_buffer_bmp(path, out.data[i].refined_a);
        std::snprintf(path, sizeof(path), "%s/refined_%02d_overlay.bmp",out_dir.c_str(), i);
        unmixing::io::write_buffer_bmp(path, out.data[i].refined_overlay);
    }
    unmixing::free_unmixed_data(out);
    std::printf("[cpu_smoke] done, wrote %d layers (6 BMPs each) under %s\n",
                out.data_num, out_dir.c_str());
    return 0;
}