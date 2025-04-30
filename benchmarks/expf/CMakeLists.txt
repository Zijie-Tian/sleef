cmake_minimum_required(VERSION 3.10)
project(sleef_expf_benchmark)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用ARM NEON支持
# 注意: 对于ARM架构这应该自动启用，此处显式设置以确保
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)")
    # 添加ARM FP16支持
    add_compile_options(-march=armv8.2-a+fp16+simd)
endif()

# 添加Google Benchmark
find_package(benchmark REQUIRED)

# 包含sleef头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../build/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../install/include)

# 链接sleef和tlfloat库位置
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../build/lib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../install/lib)

# 创建原始的float32基准测试可执行文件
add_executable(expf_benchmark expf_benchmark.cpp)

# 链接所需库
target_link_libraries(expf_benchmark 
    benchmark::benchmark
    sleef
    tlfloat
    m
    pthread)

# 创建新的float16/float32对比测试可执行文件
add_executable(expf_fp16_benchmark expf_fp16_benchmark.cpp)

# 链接所需库
target_link_libraries(expf_fp16_benchmark
    benchmark::benchmark
    sleef
    tlfloat
    m
    pthread)

# 设置优化等级
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3") 