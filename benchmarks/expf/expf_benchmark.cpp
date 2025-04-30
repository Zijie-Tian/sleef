#include <benchmark/benchmark.h>
#include <cmath>
#include <random>
#include <vector>
#include <arm_neon.h>
#include "sleef.h"

// 生成随机浮点数据
std::vector<float> generateRandomData(size_t size) {
    // 确保size是4的倍数（用于NEON向量化）
    size = (size + 3) & ~3;
    
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f); // 生成-10到10之间的随机数

    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

// 标准库 expf 性能测试（标量版本）
static void BM_StandardExpf_Scalar(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> input = generateRandomData(size);
    std::vector<float> output(size);
    
    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            output[i] = expf(input[i]);
        }
        benchmark::DoNotOptimize(output);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
    state.SetLabel("Standard expf (scalar)");
    state.counters["time_ms"] = benchmark::Counter(state.iterations() * 1000, benchmark::Counter::kAvgThreads);
}

// SLEEF expf 性能测试（标量版本）
static void BM_SleefExpf_Scalar(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> input = generateRandomData(size);
    std::vector<float> output(size);
    
    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            output[i] = Sleef_expf1_u10(input[i]);
        }
        benchmark::DoNotOptimize(output);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
    state.SetLabel("SLEEF expf_u10 (scalar)");
    state.counters["time_ms"] = benchmark::Counter(state.iterations() * 1000, benchmark::Counter::kAvgThreads);
}

// 标准库 expf 性能测试（手动向量化版本）
static void BM_StandardExpf_Manual_NEON(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> input = generateRandomData(size);
    std::vector<float> output(size);
    
    for (auto _ : state) {
        for (size_t i = 0; i < size; i += 4) {
            // 加载4个float到NEON寄存器
            float32x4_t in = vld1q_f32(&input[i]);
            
            // 手动计算每个元素的exp (无内置NEON exp函数)
            float tmp[4];
            vst1q_f32(tmp, in);  // 先存储到临时数组
            
            tmp[0] = expf(tmp[0]);
            tmp[1] = expf(tmp[1]);
            tmp[2] = expf(tmp[2]);
            tmp[3] = expf(tmp[3]);
            
            float32x4_t out = vld1q_f32(tmp);  // 再加载回NEON寄存器
            
            // 存储结果
            vst1q_f32(&output[i], out);
        }
        benchmark::DoNotOptimize(output);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
    state.SetLabel("Standard expf (manual NEON)");
    state.counters["time_ms"] = benchmark::Counter(state.iterations() * 1000, benchmark::Counter::kAvgThreads);
}

// SLEEF expf 性能测试（NEON SIMD版本）
static void BM_SleefExpf_NEON(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> input = generateRandomData(size);
    std::vector<float> output(size);
    
    for (auto _ : state) {
        for (size_t i = 0; i < size; i += 4) {
            // 加载4个float到NEON寄存器
            float32x4_t in = vld1q_f32(&input[i]);
            
            // 使用SLEEF的NEON优化版本 (4个float并行计算)
            float32x4_t out = Sleef_expf4_u10advsimd(in);
            
            // 存储结果
            vst1q_f32(&output[i], out);
        }
        benchmark::DoNotOptimize(output);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
    state.SetLabel("SLEEF expf_u10 (NEON)");
    state.counters["time_ms"] = benchmark::Counter(state.iterations() * 1000, benchmark::Counter::kAvgThreads);
}

// 注册测试
// 测试 32K 元素
BENCHMARK(BM_StandardExpf_Scalar)->Arg(32 * 1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SleefExpf_Scalar)->Arg(32 * 1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_StandardExpf_Manual_NEON)->Arg(32 * 1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SleefExpf_NEON)->Arg(32 * 1024)->Unit(benchmark::kMillisecond);

// 测试 64K 元素
BENCHMARK(BM_StandardExpf_Scalar)->Arg(64 * 1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SleefExpf_Scalar)->Arg(64 * 1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_StandardExpf_Manual_NEON)->Arg(64 * 1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SleefExpf_NEON)->Arg(64 * 1024)->Unit(benchmark::kMillisecond);

// 测试 128K 元素
BENCHMARK(BM_StandardExpf_Scalar)->Arg(128 * 1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SleefExpf_Scalar)->Arg(128 * 1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_StandardExpf_Manual_NEON)->Arg(128 * 1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SleefExpf_NEON)->Arg(128 * 1024)->Unit(benchmark::kMillisecond);

// 主函数
BENCHMARK_MAIN();