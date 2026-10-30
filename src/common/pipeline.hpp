#pragma once
#include <cstdint>
#include "backend.hpp"
#include "types.hpp"

namespace unmixing {

void run(const std::uint8_t* bgra, int w, int h,
         UnmixedData& out,
         const IBackend& backend,
         bool verbose = false);

void run(const std::uint8_t* bgra, int w, int h,
         UnmixedData& out,
         bool verbose = false);

void free_unmixed_data(UnmixedData& data);

}