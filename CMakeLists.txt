cmake_minimum_required(VERSION 3.22)

project(Lab0MatMul LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(matmul_bench
    src/main.cpp
    src/cpu_matmul.cpp
)

target_include_directories(matmul_bench PRIVATE src)
target_compile_options(matmul_bench PRIVATE -O3 -Wall -Wextra -Wpedantic)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    target_sources(matmul_bench PRIVATE src/cuda_matmul.cu)
    target_compile_definitions(matmul_bench PRIVATE HPC_HAS_CUDA=1)

    set_target_properties(matmul_bench PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_compile_options(matmul_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
else()
    message(WARNING "CUDA compiler was not found. Building CPU-only variant.")
    target_sources(matmul_bench PRIVATE src/cuda_stub.cpp)
    target_compile_definitions(matmul_bench PRIVATE HPC_HAS_CUDA=0)
endif()
