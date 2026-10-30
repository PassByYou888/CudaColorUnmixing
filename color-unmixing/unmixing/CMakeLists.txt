cmake_minimum_required(VERSION 2.8)

find_package(Eigen3 REQUIRED)
find_package(NLopt REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${NLOPT_INCLUDE_DIRS})

file(GLOB files *.cpp *.h)
add_library(ColorUnmixing STATIC ${files})
target_link_libraries(ColorUnmixing ${NLopt_LIBRARIES})