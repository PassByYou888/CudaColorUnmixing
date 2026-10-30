#pragma once
#include <string>
#include <cstdint>
#include <vector>
#include "image.hpp"
#include "types.hpp"

namespace unmixing {
namespace io {

ColorImage load_bgra(const std::uint8_t* bgra, int w, int h);

void save_gray_bgra(const GrayImage&  img, ImageBuffer& out);
void save_color_bgra(const ColorImage& img, ImageBuffer& out);

std::vector<std::uint8_t> load_bgra_from_bmp(const std::string& path, int& w, int& h);
ColorImage read_bmp(const std::string& path);

void write_bmp(const std::string& path, const ColorImage& img);
void write_bmp_gray(const std::string& path, const GrayImage& img);
void write_buffer_bmp(const std::string& path, const ImageBuffer& ib);

}
}