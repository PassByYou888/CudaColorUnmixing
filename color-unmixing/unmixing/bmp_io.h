#ifndef BMP_IO_H
#define BMP_IO_H

#include <cstdint>
#include <string>
#include <vector>

namespace BmpIO {

bool load_bmp32_bgra(const std::string& path, int& w, int& h, std::vector<std::uint8_t>& out_bgra);
bool save_bmp32_bgra(const std::string& path, int w, int h, const std::uint8_t* bgra);

}

#endif