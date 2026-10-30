#include "bmp_io.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

namespace BmpIO {

namespace {

#pragma pack(push, 1)
struct BmpFileHeader {
    std::uint16_t bfType;
    std::uint32_t bfSize;
    std::uint16_t bfReserved1;
    std::uint16_t bfReserved2;
    std::uint32_t bfOffBits;
};
struct BmpInfoHeader {
    std::uint32_t biSize;
    std::int32_t  biWidth;
    std::int32_t  biHeight;
    std::uint16_t biPlanes;
    std::uint16_t biBitCount;
    std::uint32_t biCompression;
    std::uint32_t biSizeImage;
    std::int32_t  biXPelsPerMeter;
    std::int32_t  biYPelsPerMeter;
    std::uint32_t biClrUsed;
    std::uint32_t biClrImportant;
};
#pragma pack(pop)

constexpr std::uint16_t kBmpMagic = 0x4D42;
constexpr std::uint32_t kHeaderBytes = static_cast<std::uint32_t>(sizeof(BmpFileHeader) + sizeof(BmpInfoHeader));

}

bool load_bmp32_bgra(const std::string& path, int& w, int& h, std::vector<std::uint8_t>& out_bgra)
{
    w = 0;
    h = 0;
    out_bgra.clear();

    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    BmpFileHeader fh{};
    BmpInfoHeader ih{};
    if (std::fread(&fh, sizeof(fh), 1, f) != 1 ||
        std::fread(&ih, sizeof(ih), 1, f) != 1) {
        std::fclose(f);
        return false;
    }

    if (fh.bfType != kBmpMagic) { std::fclose(f); return false; }
    if (ih.biBitCount != 32) { std::fclose(f); return false; }
    if (ih.biCompression != 0) { std::fclose(f); return false; }

    const bool top_down = (ih.biHeight < 0);
    const int W = ih.biWidth;
    const int H = top_down ? -ih.biHeight : ih.biHeight;
    if (W <= 0 || H <= 0) { std::fclose(f); return false; }

    const std::uint64_t stride64 = static_cast<std::uint64_t>(W) * 4u;
    const std::uint64_t bytes64  = stride64 * static_cast<std::uint64_t>(H);
    if (bytes64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) - kHeaderBytes) {
        std::fclose(f);
        return false;
    }
    const std::uint32_t stride = static_cast<std::uint32_t>(stride64);

    out_bgra.resize(static_cast<std::size_t>(W) * H * 4);

    if (std::fseek(f, static_cast<long>(fh.bfOffBits), SEEK_SET) != 0) {
        std::fclose(f);
        return false;
    }

    for (int file_row = 0; file_row < H; ++file_row) {
        const int y = top_down ? file_row : (H - 1 - file_row);
        std::uint8_t* dst = out_bgra.data() + static_cast<std::size_t>(y) * stride;
        if (std::fread(dst, 1, stride, f) != stride) {
            std::fclose(f);
            return false;
        }
    }
    std::fclose(f);

    bool has_alpha = false;
    for (std::size_t i = 3; i < out_bgra.size(); i += 4) {
        if (out_bgra[i] != 0) { has_alpha = true; break; }
    }
    if (!has_alpha) {
        for (std::size_t i = 3; i < out_bgra.size(); i += 4) out_bgra[i] = 255;
    }

    w = W;
    h = H;
    return true;
}

bool save_bmp32_bgra(const std::string& path, int w, int h, const std::uint8_t* bgra)
{
    if (w <= 0 || h <= 0 || !bgra) return false;

    const std::uint64_t stride64 = static_cast<std::uint64_t>(w) * 4u;
    const std::uint64_t bytes64  = stride64 * static_cast<std::uint64_t>(h);
    if (bytes64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) - kHeaderBytes) {
        return false;
    }

    const std::uint32_t data_size = static_cast<std::uint32_t>(bytes64);
    const std::uint32_t stride    = static_cast<std::uint32_t>(stride64);

    BmpFileHeader fh{};
    fh.bfType      = kBmpMagic;
    fh.bfSize      = kHeaderBytes + data_size;
    fh.bfReserved1 = 0;
    fh.bfReserved2 = 0;
    fh.bfOffBits   = kHeaderBytes;

    BmpInfoHeader ih{};
    ih.biSize          = sizeof(BmpInfoHeader);
    ih.biWidth         = w;
    ih.biHeight        = h;
    ih.biPlanes        = 1;
    ih.biBitCount      = 32;
    ih.biCompression   = 0;
    ih.biSizeImage     = data_size;
    ih.biXPelsPerMeter = 2835;
    ih.biYPelsPerMeter = 2835;
    ih.biClrUsed       = 0;
    ih.biClrImportant  = 0;

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    if (std::fwrite(&fh, sizeof(fh), 1, f) != 1 ||
        std::fwrite(&ih, sizeof(ih), 1, f) != 1) {
        std::fclose(f);
        return false;
    }

    for (int file_row = 0; file_row < h; ++file_row) {
        const int y = h - 1 - file_row;
        const std::uint8_t* src = bgra + static_cast<std::size_t>(y) * stride;
        if (std::fwrite(src, 1, stride, f) != stride) {
            std::fclose(f);
            return false;
        }
    }

    std::fclose(f);
    return true;
}

}