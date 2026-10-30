#pragma once
#include "backend.hpp"

namespace unmixing {

const IBackend& cuda_backend();
void cuda_backend_dump_stats();

}