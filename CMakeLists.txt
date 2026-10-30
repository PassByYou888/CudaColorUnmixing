cmake_minimum_required(VERSION 3.25)
project(CudaColorUnmixing VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

option(USE_CUDA   "Build CUDA backend"   OFF)
option(USE_VULKAN "Build Vulkan backend" OFF)

find_package(Eigen3 CONFIG REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/color-unmixing/cmake)

set(UNMIX_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan
)

add_library(unmixing_common STATIC
    src/common/io.cpp
    src/common/host_ops.cpp
    src/common/solver.cpp
    src/common/backend_cpu.cpp
    src/common/pipeline.cpp
)
target_include_directories(unmixing_common PUBLIC ${UNMIX_INCLUDE_DIRS})
target_link_libraries(unmixing_common PUBLIC Eigen3::Eigen)
if (MSVC)
    target_compile_options(unmixing_common PRIVATE /W4 /permissive- /utf-8 /EHsc /fp:precise)
endif()

add_executable(cpu_smoke tests/cpu_smoke/main.cpp)
target_link_libraries(cpu_smoke PRIVATE unmixing_common)
target_include_directories(cpu_smoke PRIVATE ${UNMIX_INCLUDE_DIRS})
if (MSVC)
    target_compile_options(cpu_smoke PRIVATE /W4 /permissive- /utf-8 /EHsc /fp:precise)
endif()

add_executable(bmp_smoke tests/bmp_smoke/main.cpp)
target_link_libraries(bmp_smoke PRIVATE unmixing_common)
target_include_directories(bmp_smoke PRIVATE ${UNMIX_INCLUDE_DIRS})
if (MSVC)
    target_compile_options(bmp_smoke PRIVATE /W4 /permissive- /utf-8 /EHsc /fp:precise)
endif()

if (USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "52;61")

    add_library(unmixing_cuda STATIC
        src/cuda/cuda_backend.cu
        src/cuda/backend_cuda.cu
        src/cuda/batch_solver.cu
    )
    target_include_directories(unmixing_cuda PUBLIC ${UNMIX_INCLUDE_DIRS})
    target_link_libraries(unmixing_cuda PUBLIC unmixing_common CUDA::cudart)

    target_compile_options(unmixing_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-ftz=false>
        $<$<COMPILE_LANGUAGE:CUDA>:-prec-div=true>
        $<$<COMPILE_LANGUAGE:CUDA>:-prec-sqrt=true>
    )

    add_executable(cuda_smoke tests/cuda_smoke/main.cpp)
    target_link_libraries(cuda_smoke PRIVATE unmixing_cuda)
    target_include_directories(cuda_smoke PRIVATE ${UNMIX_INCLUDE_DIRS})
    if (MSVC)
        target_compile_options(cuda_smoke PRIVATE /W4 /permissive- /utf-8 /EHsc /fp:precise)
    endif()

    add_executable(l2_verify tests/l2_verify/main.cpp)
    target_link_libraries(l2_verify PRIVATE unmixing_cuda unmixing_common)
    target_include_directories(l2_verify PRIVATE ${UNMIX_INCLUDE_DIRS})
    if (MSVC)
        target_compile_options(l2_verify PRIVATE /W4 /permissive- /utf-8 /EHsc /fp:precise)
    endif()

    add_executable(l3_verify tests/l3_verify/main.cpp)
    target_link_libraries(l3_verify PRIVATE unmixing_cuda unmixing_common)
    target_include_directories(l3_verify PRIVATE ${UNMIX_INCLUDE_DIRS})
    if (MSVC)
        target_compile_options(l3_verify PRIVATE /W4 /permissive- /utf-8 /EHsc /fp:precise)
    endif()

    add_executable(l4_verify tests/l4_verify/main.cpp)
    target_link_libraries(l4_verify PRIVATE unmixing_cuda unmixing_common)
    target_include_directories(l4_verify PRIVATE ${UNMIX_INCLUDE_DIRS})
    if (MSVC)
        target_compile_options(l4_verify PRIVATE /W4 /permissive- /utf-8 /EHsc /fp:precise)
    endif()

    target_link_libraries(bmp_smoke PRIVATE unmixing_cuda)
    target_compile_definitions(bmp_smoke PRIVATE UNMIX_HAS_CUDA=1)
endif()

if (USE_VULKAN)
    find_package(Vulkan REQUIRED)
    add_library(unmixing_vulkan STATIC src/vulkan/vulkan_backend.cpp)
    target_include_directories(unmixing_vulkan PUBLIC ${UNMIX_INCLUDE_DIRS})
    target_link_libraries(unmixing_vulkan PUBLIC unmixing_common Vulkan::Vulkan)
endif()

add_subdirectory(color-unmixing)