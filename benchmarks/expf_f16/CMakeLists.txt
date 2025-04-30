cmake_minimum_required(VERSION 3.10)
project(expf_f16_benchmark)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=armv8.2-a+fp16+simd")

# Try to find Google Benchmark package
find_package(benchmark QUIET)

# If not found via find_package, try to find it in common locations
if(NOT benchmark_FOUND)
    # Check homebrew location (macOS)
    if(EXISTS "/usr/local/include/benchmark/benchmark.h")
        set(BENCHMARK_INCLUDE_DIR "/usr/local/include")
        set(BENCHMARK_LIBRARIES "/usr/local/lib/libbenchmark.a" "/usr/local/lib/libbenchmark_main.a")
        message(STATUS "Found benchmark in homebrew location")
    # Check M1/M2 Mac homebrew location
    elseif(EXISTS "/opt/homebrew/include/benchmark/benchmark.h")
        set(BENCHMARK_INCLUDE_DIR "/opt/homebrew/include")
        set(BENCHMARK_LIBRARIES "/opt/homebrew/lib/libbenchmark.a" "/opt/homebrew/lib/libbenchmark_main.a")
        message(STATUS "Found benchmark in M1/M2 Mac homebrew location")
    else()
        message(FATAL_ERROR "Google Benchmark library not found. Please install it first.")
    endif()
else()
    message(STATUS "Found Google Benchmark via find_package")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/../../build/include
    ${CMAKE_SOURCE_DIR}/../../install/include
    ${BENCHMARK_INCLUDE_DIR}
)

# Link directories
link_directories(
    ${CMAKE_SOURCE_DIR}/../../build/lib
    ${CMAKE_SOURCE_DIR}/../../install/lib
)

# Add executable
add_executable(expf_f16_benchmark expf_f16_benchmark.cpp)

# Link libraries
if(benchmark_FOUND)
    target_link_libraries(expf_f16_benchmark
        sleef
        tlfloat
        m
        benchmark::benchmark
        pthread
    )
else()
    target_link_libraries(expf_f16_benchmark
        sleef
        tlfloat
        m
        ${BENCHMARK_LIBRARIES}
        pthread
    )
endif()

# Add custom target for running the benchmark
add_custom_target(run_benchmark
    COMMAND ./expf_f16_benchmark
    DEPENDS expf_f16_benchmark
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
) 