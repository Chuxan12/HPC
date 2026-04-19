cmake_minimum_required(VERSION 3.22)

project(Lab2GeneticAlgorithm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(WARNING "Recommended compiler for this lab: clang/clang++. Current: ${CMAKE_CXX_COMPILER_ID}")
endif()

add_executable(genetic_algorithm_bench
    src/main.cpp
    src/genetic_algorithm.cpp
    src/genetic_algorithm_cuda.cu
)

target_include_directories(genetic_algorithm_bench PRIVATE src)
target_compile_options(genetic_algorithm_bench PRIVATE -O3 -Wall -Wextra -Wpedantic)

set_target_properties(genetic_algorithm_bench PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(genetic_algorithm_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_link_libraries(genetic_algorithm_bench PRIVATE curand)
