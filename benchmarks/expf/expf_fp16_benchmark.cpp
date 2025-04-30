#include <benchmark/benchmark.h>
#include <cmath>
#include <random>
#include <vector>
#include <arm_neon.h>
#include "sleef.h"

// 使用__fp16类型表示float16
#ifdef __ARM_FP16_FORMAT_IEEE
typedef __fp16 float16_t;
#else
// 如果编译器不支持__fp16，使用uint16_t来存储half-precision浮点数
typedef uint16_t float16_t;
#endif

// 生成随机float32数据
std::vector<float> generateRandomDataF32(size_t size) {
    // 确保size是4的倍数（用于NEON向量化）
    size = (size + 3) & ~3;
    
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

// 转换float32到float16
std::vector<float16_t> convertF32ToF16(const std::vector<float>& f32_data) {
    std::vector<float16_t> f16_data(f32_data.size());
    
    for (size_t i = 0; i < f32_data.size(); i += 4) {
        float32x4_t f32x4 = vld1q_f32(&f32_data[i]);
        
        // 转换为float16
        #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        float16x4_t f16x4 = vcvt_f16_f32(f32x4);
        vst1_f16((float16_t*)&f16_data[i], f16x4);
        #else
        // 手动转换，当没有FP16 vector支持时
        for (int j = 0; j < 4 && (i + j) < f32_data.size(); ++j) {
            // IEEE-754 半精度浮点数转换
            // 简化实现，实际应用中应使用更精确的转换
            uint32_t x = *(uint32_t*)&f32_data[i + j];
            uint16_t h = ((x >> 16) & 0x8000) | // 符号位
                         ((((x >> 23) & 0xff) - 127 + 15) << 10) | // 指数
                         ((x >> 13) & 0x3ff); // 尾数
            f16_data[i + j] = *(float16_t*)&h;
        }
        #endif
    }
    
    return f16_data;
}

// 转换float16到float32
std::vector<float> convertF16ToF32(const std::vector<float16_t>& f16_data) {
    std::vector<float> f32_data(f16_data.size());
    
    for (size_t i = 0; i < f16_data.size(); i += 4) {
        // 转换为float32
        #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        float16x4_t f16x4 = vld1_f16((float16_t*)&f16_data[i]);
        float32x4_t f32x4 = vcvt_f32_f16(f16x4);
        vst1q_f32(&f32_data[i], f32x4);
        #else
        // 手动转换，当没有FP16 vector支持时
        for (int j = 0; j < 4 && (i + j) < f16_data.size(); ++j) {
            // IEEE-754 半精度浮点数转换
            uint16_t h = *(uint16_t*)&f16_data[i + j];
            uint32_t x = ((h & 0x8000) << 16) | // 符号位
                         (((((h >> 10) & 0x1f) - 15 + 127) << 23)) | // 指数
                         ((h & 0x3ff) << 13); // 尾数
            f32_data[i + j] = *(float*)&x;
        }
        #endif
    }
    
    return f32_data;
}

// ================ Float32 测试 ================

// 标准库 expf (float32) 性能测试
static void BM_StandardExp_F32(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> input = generateRandomDataF32(size);
    std::vector<float> output(size);
    
    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            output[i] = expf(input[i]);
        }
        benchmark::DoNotOptimize(output);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
    state.SetLabel("Standard expf (F32)");
    state.counters["time_ms"] = benchmark::Counter(state.iterations() * 1000, benchmark::Counter::kAvgThreads);
}

// SLEEF expf (float32) 标量版本性能测试
static void BM_SleefExp_F32_Scalar(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> input = generateRandomDataF32(size);
    std::vector<float> output(size);
    
    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            output[i] = Sleef_expf1_u10(input[i]);
        }
        benchmark::DoNotOptimize(output);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
    state.SetLabel("SLEEF expf_u10 (F32 scalar)");
    state.counters["time_ms"] = benchmark::Counter(state.iterations() * 1000, benchmark::Counter::kAvgThreads);
}

// SLEEF expf (float32) NEON版本性能测试
static void BM_SleefExp_F32_NEON(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> input = generateRandomDataF32(size);
    std::vector<float> output(size);
    
    for (auto _ : state) {
        for (size_t i = 0; i < size; i += 4) {
            float32x4_t in = vld1q_f32(&input[i]);
            float32x4_t out = Sleef_expf4_u10advsimd(in);
            vst1q_f32(&output[i], out);
        }
        benchmark::DoNotOptimize(output);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
    state.SetLabel("SLEEF expf_u10 (F32 NEON)");
    state.counters["time_ms"] = benchmark::Counter(state.iterations() * 1000, benchmark::Counter::kAvgThreads);
}

// ================ Float16 测试 ================

// 标准库 expf (float16) 性能测试 - 使用转换
static void BM_StandardExp_F16(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> f32_input = generateRandomDataF32(size);
    std::vector<float16_t> f16_input = convertF32ToF16(f32_input);
    std::vector<float16_t> f16_output(size);
    
    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            // 转换为float32，计算，再转回float16
            float f32_val = (float)f16_input[i];
            f16_output[i] = (float16_t)expf(f32_val);
        }
        benchmark::DoNotOptimize(f16_output);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float16_t));
    state.SetLabel("Standard expf (F16 via F32)");
    state.counters["time_ms"] = benchmark::Counter(state.iterations() * 1000, benchmark::Counter::kAvgThreads);
}

// SLEEF expf (float16) 性能测试 - 使用转换
static void BM_SleefExp_F16_Scalar(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> f32_input = generateRandomDataF32(size);
    std::vector<float16_t> f16_input = convertF32ToF16(f32_input);
    std::vector<float16_t> f16_output(size);
    
    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            // 转换为float32，计算，再转回float16
            float f32_val = (float)f16_input[i];
            f16_output[i] = (float16_t)Sleef_expf1_u10(f32_val);
        }
        benchmark::DoNotOptimize(f16_output);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float16_t));
    state.SetLabel("SLEEF expf_u10 (F16 via F32 scalar)");
    state.counters["time_ms"] = benchmark::Counter(state.iterations() * 1000, benchmark::Counter::kAvgThreads);
}

// SLEEF expf (float16) NEON版本性能测试 - 使用转换
static void BM_SleefExp_F16_NEON(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> f32_input = generateRandomDataF32(size);
    std::vector<float16_t> f16_input = convertF32ToF16(f32_input);
    std::vector<float16_t> f16_output(size);
    
    for (auto _ : state) {
        for (size_t i = 0; i < size; i += 4) {
            // 转换为float32
            #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            float16x4_t f16x4 = vld1_f16((float16_t*)&f16_input[i]);
            float32x4_t f32x4 = vcvt_f32_f16(f16x4);
            
            // 计算exp
            float32x4_t result_f32 = Sleef_expf4_u10advsimd(f32x4);
            
            // 转回float16
            float16x4_t result_f16 = vcvt_f16_f32(result_f32);
            vst1_f16((float16_t*)&f16_output[i], result_f16);
            #else
            // 手动转换和计算
            float tmp_in[4], tmp_out[4];
            
            // 转换为float32
            for (int j = 0; j < 4 && (i + j) < size; ++j) {
                tmp_in[j] = (float)f16_input[i + j];
            }
            
            // 加载、计算、存储
            float32x4_t in = vld1q_f32(tmp_in);
            float32x4_t out = Sleef_expf4_u10advsimd(in);
            vst1q_f32(tmp_out, out);
            
            // 转回float16
            for (int j = 0; j < 4 && (i + j) < size; ++j) {
                f16_output[i + j] = (float16_t)tmp_out[j];
            }
            #endif
        }
        benchmark::DoNotOptimize(f16_output);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float16_t));
    state.SetLabel("SLEEF expf_u10 (F16 via F32 NEON)");
    state.counters["time_ms"] = benchmark::Counter(state.iterations() * 1000, benchmark::Counter::kAvgThreads);
}

// 注册测试 - 只测试128K大小
BENCHMARK(BM_StandardExp_F32)->Arg(128 * 1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SleefExp_F32_Scalar)->Arg(128 * 1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SleefExp_F32_NEON)->Arg(128 * 1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_StandardExp_F16)->Arg(128 * 1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SleefExp_F16_Scalar)->Arg(128 * 1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SleefExp_F16_NEON)->Arg(128 * 1024)->Unit(benchmark::kMillisecond);

// 主函数
BENCHMARK_MAIN(); 