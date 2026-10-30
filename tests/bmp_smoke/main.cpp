#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "pipeline.hpp"
#include "io.hpp"
#include "backend_cpu.hpp"

#ifdef UNMIX_HAS_CUDA
#include "backend_cuda.hpp"
#include "cuda_backend.hpp"
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <input.bmp> [output_dir] [cpu|cuda] [--swap-rb]\n", argv[0]);
        std::fprintf(stderr, "  output_dir default: <input_stem>_out\n");
        std::fprintf(stderr, "  backend    default: cuda if available, else cpu\n");
        std::fprintf(stderr, "  --swap-rb  swap R and B channels after load (diagnostic)\n");
        return 1;
    }

    const std::string input = argv[1];
    std::string out_dir;
    std::string backend_name;
    bool swap_rb = false;

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--swap-rb") == 0) { swap_rb = true; continue; }
        if (out_dir.empty()) out_dir = argv[i];
        else if (backend_name.empty()) backend_name = argv[i];
    }

    if (out_dir.empty()) {
        std::filesystem::path p(input);
        out_dir = (p.parent_path() / (p.stem().string() + "_out")).string();
    }

    int w = 0, h = 0;
    std::vector<std::uint8_t> bgra = unmixing::io::load_bgra_from_bmp(input, w, h);
    if (bgra.empty() || w <= 0 || h <= 0) {
        std::fprintf(stderr, "[bmp_smoke] failed to load '%s'\n", input.c_str());
        std::fprintf(stderr, "[bmp_smoke] supported: 32-bit BI_RGB BMP (no RLE, no BITFIELDS, no palette)\n");
        return 2;
    }

    std::printf("[bmp_smoke] loaded '%s': %d x %d\n", input.c_str(), w, h);

    std::printf("[probe] first 5 pixels raw BGRA bytes:\n");
    const int probe_n = std::min(5, w * h);
    for (int i = 0; i < probe_n; ++i) {
        std::printf("  pixel[%d]: B=%3u G=%3u R=%3u A=%3u\n",
                    i, bgra[(std::size_t)i*4+0], bgra[(std::size_t)i*4+1],
                    bgra[(std::size_t)i*4+2], bgra[(std::size_t)i*4+3]);
    }

    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        std::fprintf(stderr, "[bmp_smoke] create_directories('%s') failed: %s\n",
                     out_dir.c_str(), ec.message().c_str());
        return 3;
    }

    {
        unmixing::ColorImage rt = unmixing::io::load_bgra(bgra.data(), w, h);
        if (swap_rb) {
            for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
                const double tmp = rt.r().at(x, y);
                rt.r().at(x, y) = rt.b().at(x, y);
                rt.b().at(x, y) = tmp;
            }
        }
        std::string rtpath = out_dir + "/roundtrip.bmp";
        unmixing::io::write_bmp(rtpath, rt);
        std::printf("[probe] wrote %s (open in Paint and compare to original)\n", rtpath.c_str());
    }

    if (swap_rb) {
        for (std::size_t i = 0; i < bgra.size(); i += 4) std::swap(bgra[i], bgra[i+2]);
        std::printf("[bmp_smoke] --swap-rb: R and B swapped in input\n");
    }

    const unmixing::IBackend* backend = nullptr;
    std::string used;

#ifdef UNMIX_HAS_CUDA
    if (backend_name.empty() || backend_name == "cuda") {
        if (unmixing::cuda::available()) {
            backend = &unmixing::cuda_backend();
            used = "cuda";
        } else if (backend_name == "cuda") {
            std::fprintf(stderr, "[bmp_smoke] cuda requested but no device available\n");
            return 4;
        }
    }
#endif

    if (!backend) {
        if (backend_name == "cuda") {
            std::fprintf(stderr, "[bmp_smoke] cuda requested but not compiled in (USE_CUDA=OFF)\n");
            return 4;
        }
        backend = &unmixing::cpu_backend();
        used = "cpu";
    }
    std::printf("[bmp_smoke] backend = %s\n", used.c_str());

    unmixing::UnmixedData out;
    unmixing::run(bgra.data(), w, h, out, *backend, true);

    if (out.data_num <= 0) {
        std::fprintf(stderr, "[bmp_smoke] no layers produced\n");
        unmixing::free_unmixed_data(out);
        return 5;
    }

    char path[2048];
    for (int i = 0; i < out.data_num; ++i) {
        std::snprintf(path, sizeof(path), "%s/layer_%02d_color.bmp",     out_dir.c_str(), i);
        unmixing::io::write_buffer_bmp(path, out.data[i].output);
        std::snprintf(path, sizeof(path), "%s/layer_%02d_alpha.bmp",     out_dir.c_str(), i);
        unmixing::io::write_buffer_bmp(path, out.data[i].output_a);
        std::snprintf(path, sizeof(path), "%s/layer_%02d_overlay.bmp",   out_dir.c_str(), i);
        unmixing::io::write_buffer_bmp(path, out.data[i].overlay);
        std::snprintf(path, sizeof(path), "%s/refined_%02d_color.bmp",   out_dir.c_str(), i);
        unmixing::io::write_buffer_bmp(path, out.data[i].refined);
        std::snprintf(path, sizeof(path), "%s/refined_%02d_alpha.bmp",   out_dir.c_str(), i);
        unmixing::io::write_buffer_bmp(path, out.data[i].refined_a);
        std::snprintf(path, sizeof(path), "%s/refined_%02d_overlay.bmp", out_dir.c_str(), i);
        unmixing::io::write_buffer_bmp(path, out.data[i].refined_overlay);
    }

#ifdef UNMIX_HAS_CUDA
    if (used == "cuda") {
        unmixing::cuda_backend_dump_stats();
    }
#endif

    std::printf("[bmp_smoke] done, wrote %d layers (6 BMPs each) under '%s'\n",
                out.data_num, out_dir.c_str());
    unmixing::free_unmixed_data(out);
    return 0;
}