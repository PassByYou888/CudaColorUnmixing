#include <string>
#include <filesystem>
#include <system_error>
#include <cstdio>
#include "unmixing.h"

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <input.bmp> <output_dir>\n", argv[0]);
        return 1;
    }
    const std::string image_file_path(argv[1]);
    const std::string output_directory_path(argv[2]);

    std::error_code ec;
    std::filesystem::create_directories(output_directory_path, ec);
    if (ec) {
        std::fprintf(stderr, "failed to create output dir: %s\n", ec.message().c_str());
        return 2;
    }

    ColorUnmixing::compute_color_unmixing(image_file_path, output_directory_path);

    return 0;
}