find_package(NLopt REQUIRED)

file(GLOB UNMIXING_REF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/unmixing/*.cpp)

add_library(ColorUnmixingRef STATIC ${UNMIXING_REF_SRC})
target_include_directories(ColorUnmixingRef PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/unmixing
    ${EIGEN3_INCLUDE_DIR}
    ${NLOPT_INCLUDE_DIRS}
)
target_link_libraries(ColorUnmixingRef PUBLIC Eigen3::Eigen ${NLOPT_LIBRARIES})
if (MSVC)
    target_compile_options(ColorUnmixingRef PRIVATE /utf-8 /EHsc)
endif()

add_executable(color-unmixing test.cpp)
target_link_libraries(color-unmixing PRIVATE ColorUnmixingRef)
if (MSVC)
    target_compile_options(color-unmixing PRIVATE /utf-8 /EHsc)
endif()