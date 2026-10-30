#define _CRT_SECURE_NO_WARNINGS
#include "io.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>

namespace unmixing {
namespace io {

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

static_assert(sizeof(BmpFileHeader) == 14, "BmpFileHeader size must be 14");
static_assert(sizeof(BmpInfoHeader) == 40, "BmpInfoHeader size must be 40");

constexpr std::uint16_t kBmpMagic = 0x4D42;
constexpr std::uint32_t kHeaderBytes =
    static_cast<std::uint32_t>(sizeof(BmpFileHeader) + sizeof(BmpInfoHeader));

std::uint32_t row_stride_32(int width) {
    return static_cast<std::uint32_t>(width) * 4u;
}

bool check_data_size(int W, int H, std::uint32_t& out_data_size, std::uint32_t& out_stride) {
    const std::uint64_t stride = static_cast<std::uint64_t>(W) * 4u;
    const std::uint64_t bytes  = stride * static_cast<std::uint64_t>(H);
    if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) - kHeaderBytes) {
        std::fprintf(stderr, "[io] BMP too large to fit in uint32 (%d x %d)\n", W, H);
        return false;
    }
    out_stride    = static_cast<std::uint32_t>(stride);
    out_data_size = static_cast<std::uint32_t>(bytes);
    return true;
}

bool read_bmp_headers(FILE* f, BmpFileHeader& fh, BmpInfoHeader& ih,
                      const char* path, int& W, int& H, bool& top_down) {
    if (std::fread(&fh, sizeof(fh), 1, f) != 1 ||
        std::fread(&ih, sizeof(ih), 1, f) != 1) {
        std::fprintf(stderr, "[io] failed to read BMP headers: %s\n", path);
        return false;
    }
    if (fh.bfType != kBmpMagic) {
        std::fprintf(stderr, "[io] not a BMP (magic 0x%04x): %s\n",
                     static_cast<unsigned>(fh.bfType), path);
        return false;
    }
    if (ih.biBitCount != 32) {
        std::fprintf(stderr, "[io] BMP must be 32-bit, got %u: %s\n",
                     static_cast<unsigned>(ih.biBitCount), path);
        return false;
    }
    if (ih.biCompression != 0) {
        std::fprintf(stderr, "[io] BMP must be BI_RGB (0), got %u: %s\n",
                     static_cast<unsigned>(ih.biCompression), path);
        return false;
    }
    if (ih.biWidth <= 0 || ih.biHeight == 0) {
        std::fprintf(stderr, "[io] invalid BMP dimensions %d x %d: %s\n",
                     ih.biWidth, ih.biHeight, path);
        return false;
    }
    top_down = (ih.biHeight < 0);
    W = ih.biWidth;
    H = top_down ? -ih.biHeight : ih.biHeight;
    return true;
}

bool open_bmp_for_write(FILE*& f, const std::string& path, const char* what) {
    f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "[io] failed to open %s for write: %s\n", what, path.c_str());
        return false;
    }
    return true;
}

bool write_bmp_headers(FILE* f, int W, int H, std::uint32_t data_size) {
    BmpFileHeader fh{};
    fh.bfType      = kBmpMagic;
    fh.bfSize      = kHeaderBytes + data_size;
    fh.bfReserved1 = 0;
    fh.bfReserved2 = 0;
    fh.bfOffBits   = kHeaderBytes;

    BmpInfoHeader ih{};
    ih.biSize          = sizeof(BmpInfoHeader);
    ih.biWidth         = W;
    ih.biHeight        = H;
    ih.biPlanes        = 1;
    ih.biBitCount      = 32;
    ih.biCompression   = 0;
    ih.biSizeImage     = data_size;
    ih.biXPelsPerMeter = 2835;
    ih.biYPelsPerMeter = 2835;
    ih.biClrUsed       = 0;
    ih.biClrImportant  = 0;

    return std::fwrite(&fh, sizeof(fh), 1, f) == 1 &&
           std::fwrite(&ih, sizeof(ih), 1, f) == 1;
}

bool write_bgra_rows_bottom_up(FILE* f, const std::uint8_t* src, int W, int H) {
    const std::uint32_t stride = row_stride_32(W);
    for (int file_row = 0; file_row < H; ++file_row) {
        const int y = H - 1 - file_row;
        const std::uint8_t* srow = src + static_cast<std::size_t>(y) * stride;
        if (std::fwrite(srow, 1, stride, f) != stride) {
            std::fprintf(stderr, "[io] short write (row %d)\n", file_row);
            return false;
        }
    }
    return true;
}

}

ColorImage load_bgra(const std::uint8_t* bgra, int w, int h) {
    ColorImage img(w, h);
    for (int y = 0; y < h; ++y) {
        double* dr = img.r().row(y);
        double* dg = img.g().row(y);
        double* db = img.b().row(y);
        double* da = img.a().row(y);
        for (int x = 0; x < w; ++x) {
            const std::uint8_t* p = bgra + (static_cast<std::size_t>(y) * w + x) * 4;
            db[x] = p[0] / 255.0;
            dg[x] = p[1] / 255.0;
            dr[x] = p[2] / 255.0;
            da[x] = p[3] / 255.0;
        }
    }
    return img;
}

void save_gray_bgra(const GrayImage& img, ImageBuffer& out) {
    const int W = img.width(), H = img.height();
    out.width  = W;
    out.height = H;
    out.bits   = std::malloc(sizeof(std::uint8_t) * 4 * static_cast<std::size_t>(W) * H);
    if (!out.bits) {
        std::fprintf(stderr, "[io] malloc failed for %d x %d gray buffer\n", W, H);
        out.width = 0;
        out.height = 0;
        return;
    }
    std::uint8_t* p = static_cast<std::uint8_t*>(out.bits);
    for (int y = 0; y < H; ++y) {
        const double* row = img.row(y);
        for (int x = 0; x < W; ++x) {
            const std::uint8_t u =
                static_cast<std::uint8_t>(std::clamp(row[x], 0.0, 1.0) * 255.0 + 0.5);
            p[0] = u; p[1] = u; p[2] = u; p[3] = 255;
            p += 4;
        }
    }
}

void save_color_bgra(const ColorImage& img, ImageBuffer& out) {
    const int W = img.width(), H = img.height();
    out.width  = W;
    out.height = H;
    out.bits   = std::malloc(sizeof(std::uint8_t) * 4 * static_cast<std::size_t>(W) * H);
    if (!out.bits) {
        std::fprintf(stderr, "[io] malloc failed for %d x %d color buffer\n", W, H);
        out.width = 0;
        out.height = 0;
        return;
    }
    std::uint8_t* p = static_cast<std::uint8_t*>(out.bits);
    for (int y = 0; y < H; ++y) {
        const double* rr = img.r().row(y);
        const double* gg = img.g().row(y);
        const double* bb = img.b().row(y);
        const double* aa = img.a().row(y);
        for (int x = 0; x < W; ++x) {
            p[0] = static_cast<std::uint8_t>(std::clamp(bb[x], 0.0, 1.0) * 255.0 + 0.5);
            p[1] = static_cast<std::uint8_t>(std::clamp(gg[x], 0.0, 1.0) * 255.0 + 0.5);
            p[2] = static_cast<std::uint8_t>(std::clamp(rr[x], 0.0, 1.0) * 255.0 + 0.5);
            p[3] = static_cast<std::uint8_t>(std::clamp(aa[x], 0.0, 1.0) * 255.0 + 0.5);
            p += 4;
        }
    }
}

std::vector<std::uint8_t> load_bgra_from_bmp(const std::string& path, int& w, int& h) {
    w = 0; h = 0;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "[io] failed to open BMP for read: %s\n", path.c_str());
        return {};
    }
    BmpFileHeader fh{};
    BmpInfoHeader ih{};
    int W = 0, H = 0;
    bool top_down = false;
    if (!read_bmp_headers(f, fh, ih, path.c_str(), W, H, top_down)) {
        std::fclose(f);
        return {};
    }
    std::uint32_t data_size = 0, stride = 0;
    if (!check_data_size(W, H, data_size, stride)) {
        std::fclose(f);
        return {};
    }
    (void)data_size;

    std::vector<std::uint8_t> bgra(static_cast<std::size_t>(W) * H * 4);
    if (std::fseek(f, static_cast<long>(fh.bfOffBits), SEEK_SET) != 0) {
        std::fprintf(stderr, "[io] fseek failed: %s\n", path.c_str());
        std::fclose(f);
        return {};
    }
    for (int file_row = 0; file_row < H; ++file_row) {
        const int y = top_down ? file_row : (H - 1 - file_row);
        std::uint8_t* drow = bgra.data() + static_cast<std::size_t>(y) * stride;
        if (std::fread(drow, 1, stride, f) != stride) {
            std::fprintf(stderr, "[io] truncated BMP data (row %d): %s\n",
                         file_row, path.c_str());
            std::fclose(f);
            return {};
        }
    }
    std::fclose(f);

    bool has_alpha = false;
    for (std::size_t i = 3; i < bgra.size(); i += 4) {
        if (bgra[i] != 0) { has_alpha = true; break; }
    }
    if (!has_alpha) {
        std::fprintf(stderr, "[io] BMP has all-zero alpha, forcing 255 (RGB-only file): %s\n",
                     path.c_str());
        for (std::size_t i = 3; i < bgra.size(); i += 4) bgra[i] = 255;
    }

    w = W; h = H;
    return bgra;
}

ColorImage read_bmp(const std::string& path) {
    int w = 0, h = 0;
    auto bgra = load_bgra_from_bmp(path, w, h);
    if (bgra.empty() || w == 0 || h == 0) return ColorImage();
    return load_bgra(bgra.data(), w, h);
}

void write_bmp(const std::string& path, const ColorImage& img) {
    const int W = img.width(), H = img.height();
    if (W <= 0 || H <= 0) return;
    std::uint32_t data_size = 0, stride = 0;
    if (!check_data_size(W, H, data_size, stride)) return;

    FILE* f = nullptr;
    if (!open_bmp_for_write(f, path, "bmp")) return;
    if (!write_bmp_headers(f, W, H, data_size)) {
        std::fprintf(stderr, "[io] failed to write BMP headers: %s\n", path.c_str());
        std::fclose(f);
        return;
    }

    std::vector<std::uint8_t> row(stride);
    for (int file_row = 0; file_row < H; ++file_row) {
        const int y = H - 1 - file_row;
        const double* rr = img.r().row(y);
        const double* gg = img.g().row(y);
        const double* bb = img.b().row(y);
        const double* aa = img.a().row(y);
        std::uint8_t* d = row.data();
        for (int x = 0; x < W; ++x) {
            d[0] = static_cast<std::uint8_t>(std::clamp(bb[x], 0.0, 1.0) * 255.0 + 0.5);
            d[1] = static_cast<std::uint8_t>(std::clamp(gg[x], 0.0, 1.0) * 255.0 + 0.5);
            d[2] = static_cast<std::uint8_t>(std::clamp(rr[x], 0.0, 1.0) * 255.0 + 0.5);
            d[3] = static_cast<std::uint8_t>(std::clamp(aa[x], 0.0, 1.0) * 255.0 + 0.5);
            d += 4;
        }
        if (std::fwrite(row.data(), 1, stride, f) != stride) {
            std::fprintf(stderr, "[io] short write (row %d): %s\n", file_row, path.c_str());
            std::fclose(f);
            return;
        }
    }
    std::fclose(f);
}

void write_bmp_gray(const std::string& path, const GrayImage& img) {
    const int W = img.width(), H = img.height();
    if (W <= 0 || H <= 0) return;
    std::uint32_t data_size = 0, stride = 0;
    if (!check_data_size(W, H, data_size, stride)) return;

    FILE* f = nullptr;
    if (!open_bmp_for_write(f, path, "bmp-gray")) return;
    if (!write_bmp_headers(f, W, H, data_size)) {
        std::fprintf(stderr, "[io] failed to write BMP headers: %s\n", path.c_str());
        std::fclose(f);
        return;
    }

    std::vector<std::uint8_t> row(stride);
    for (int file_row = 0; file_row < H; ++file_row) {
        const int y = H - 1 - file_row;
        const double* v = img.row(y);
        std::uint8_t* d = row.data();
        for (int x = 0; x < W; ++x) {
            const std::uint8_t u =
                static_cast<std::uint8_t>(std::clamp(v[x], 0.0, 1.0) * 255.0 + 0.5);
            d[0] = u; d[1] = u; d[2] = u; d[3] = 255;
            d += 4;
        }
        if (std::fwrite(row.data(), 1, stride, f) != stride) {
            std::fprintf(stderr, "[io] short write (row %d): %s\n", file_row, path.c_str());
            std::fclose(f);
            return;
        }
    }
    std::fclose(f);
}

void write_buffer_bmp(const std::string& path, const ImageBuffer& ib) {
    const int W = ib.width, H = ib.height;
    if (W <= 0 || H <= 0 || !ib.bits) {
        std::fprintf(stderr, "[io] write_buffer_bmp: empty buffer: %s\n", path.c_str());
        return;
    }
    std::uint32_t data_size = 0, stride = 0;
    if (!check_data_size(W, H, data_size, stride)) return;

    FILE* f = nullptr;
    if (!open_bmp_for_write(f, path, "bmp-buffer")) return;
    if (!write_bmp_headers(f, W, H, data_size)) {
        std::fprintf(stderr, "[io] failed to write BMP headers: %s\n", path.c_str());
        std::fclose(f);
        return;
    }

    const std::uint8_t* src = static_cast<const std::uint8_t*>(ib.bits);
    if (!write_bgra_rows_bottom_up(f, src, W, H)) {
        std::fprintf(stderr, "[io] failed to write BMP data: %s\n", path.c_str());
        std::fclose(f);
        return;
    }
    std::fclose(f);
}

}
}