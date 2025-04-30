#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <arm_neon.h>
#include "sleef.h"

// 使用__fp16类型表示float16
#ifdef __ARM_FP16_FORMAT_IEEE
typedef __fp16 float16_t;
#else
// 如果编译器不支持__fp16，使用uint16_t来存储half-precision浮点数
typedef uint16_t float16_t;
#endif

//====================== 自定义NEON Float16 Exp函数 ======================

// Helper constants (pre-calculated for float16)
#define LOG2_E_F16  ((float16_t)1.442695f)  // log2(e)
#define LN2_F16     ((float16_t)0.693147f)  // ln(2)
#define HALF_F16    ((float16_t)0.5f)

// Polynomial coefficients for exp(r) where r is small
#define C0_F16 ((float16_t)1.0f)
#define C1_F16 ((float16_t)1.0f)
#define C2_F16 ((float16_t)0.5f)         // 1/2!
#define C3_F16 ((float16_t)0.16667f)    // 1/3! (approx)
#define C4_F16 ((float16_t)0.04167f)    // 1/4! (approx)
#define C5_F16 ((float16_t)0.00833f)    // 1/5! (approx)

// Input thresholds for float16 expf to avoid overflow/underflow
#define EXP_MAX_INPUT_F16 ((float16_t)11.09f)
#define EXP_MIN_INPUT_F16 ((float16_t)-10.0f)

/**
 * @brief Calculates the exponential function (e^x) for four float16 values using NEON.
 *
 * @param x Input vector of 4 float16 values.
 * @return float16x4_t Vector containing exp(x) for each element.
 */
inline float16x4_t expf_neon_f16x4(float16x4_t x) {
    // --- Constants ---
    const float16x4_t log2_e = vdup_n_f16(LOG2_E_F16);
    const float16x4_t ln2    = vdup_n_f16(LN2_F16);
    const float16x4_t half   = vdup_n_f16(HALF_F16);

    const float16x4_t c0 = vdup_n_f16(C0_F16);
    const float16x4_t c1 = vdup_n_f16(C1_F16);
    const float16x4_t c2 = vdup_n_f16(C2_F16);
    const float16x4_t c3 = vdup_n_f16(C3_F16);
    const float16x4_t c4 = vdup_n_f16(C4_F16);
    const float16x4_t c5 = vdup_n_f16(C5_F16);

    const float16x4_t max_input = vdup_n_f16(EXP_MAX_INPUT_F16);
    const float16x4_t min_input = vdup_n_f16(EXP_MIN_INPUT_F16);
    const float16x4_t infinity  = vdup_n_f16((float16_t)INFINITY);
    const float16x4_t zero      = vdup_n_f16((float16_t)0.0f);

    // --- Range Reduction ---
    // Calculate n = round(x / ln(2)) = round(x * log2(e))
    float16x4_t v = vmul_f16(x, log2_e);
    
    // 转换为float32来进行整数转换（解决vcvtq_s32_f16缺失问题）
    float32x4_t v_f32 = vcvt_f32_f16(v);
    int32x4_t n_s32 = vcvtq_s32_f32(v_f32);

    // 使用float32作为中间类型（解决vcvt_f16_s32缺失问题）
    float32x4_t n_f32 = vcvtq_f32_s32(n_s32);
    float16x4_t n_f16 = vcvt_f16_f32(n_f32);

    // Calculate remainder r = x - n * ln(2)
    float16x4_t r = vfms_f16(x, n_f16, ln2); // r = x - n_f16 * ln2

    // --- Polynomial Approximation for exp(r) ---
    float16x4_t poly = vfma_f16(c4, r, c5);     // poly = c4 + r*c5
    poly = vfma_f16(c3, r, poly);               // poly = c3 + r*(c4 + r*c5)
    poly = vfma_f16(c2, r, poly);               // poly = c2 + r*(...)
    poly = vfma_f16(c1, r, poly);               // poly = c1 + r*(...)
    poly = vfma_f16(c0, r, poly);               // poly = c0 + r*(...) -> This is P(r) approx exp(r)

    // --- Scaling by 2^n ---
    // 使用手动方法实现指数调整，避免使用位操作
    // 将多项式结果和n转换为float32以便进行精确计算
    float32x4_t poly_f32 = vcvt_f32_f16(poly);
    float32x4_t scale_f32;
    
    // 计算2^n的值（分别计算每个元素）
    float tmp_n[4], tmp_scale[4];
    vst1q_f32(tmp_n, n_f32);  // 将n存储到临时数组
    
    for (int i = 0; i < 4; i++) {
        tmp_scale[i] = std::exp2f(tmp_n[i]);  // 使用标准库的exp2计算2^n
    }
    
    scale_f32 = vld1q_f32(tmp_scale);  // 加载计算结果
    
    // 计算结果 = poly * 2^n
    float32x4_t result_f32 = vmulq_f32(poly_f32, scale_f32);
    float16x4_t result = vcvt_f16_f32(result_f32);

    // --- Handle Overflow/Underflow ---
    uint16x4_t overflow_mask  = vcgt_f16(x, max_input); // x > max_input ? 1s : 0s
    uint16x4_t underflow_mask = vclt_f16(x, min_input); // x < min_input ? 1s : 0s

    // Use bit select (vbsl) to merge results
    result = vbsl_f16(overflow_mask, infinity, result);
    result = vbsl_f16(underflow_mask, zero, result);

    return result;
}

//====================== 辅助函数 ======================

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
        
        #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        float16x4_t f16x4 = vcvt_f16_f32(f32x4);
        vst1_f16((float16_t*)&f16_data[i], f16x4);
        #else
        // 手动转换
        for (int j = 0; j < 4 && (i + j) < f32_data.size(); ++j) {
            f16_data[i + j] = (float16_t)f32_data[i + j];
        }
        #endif
    }
    
    return f16_data;
}

//====================== 基准测试函数 ======================

// 自定义float16 NEON expf 性能测试
static void BM_Custom_ExpF16_NEON(benchmark::State& state) {
    const size_t size = state.range(0);
    
    // 生成随机数据
    std::vector<float> f32_input = generateRandomDataF32(size);
    std::vector<float16_t> f16_input = convertF32ToF16(f32_input);
    std::vector<float16_t> f16_output(size);
    
    for (auto _ : state) {
        // 使用自定义实现的函数计算
        for (size_t i = 0; i < size; i += 4) {
            float16x4_t input = vld1_f16((float16_t*)&f16_input[i]);
            float16x4_t result = expf_neon_f16x4(input);
            vst1_f16((float16_t*)&f16_output[i], result);
        }
        
        benchmark::DoNotOptimize(f16_output);
        benchmark::ClobberMemory();
    }
    
    // 设置性能指标
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float16_t));
    state.SetLabel("Custom FP16 NEON");
}

// SLEEF的float16 (通过float32实现) 性能测试
static void BM_SLEEF_ExpF16_via_F32(benchmark::State& state) {
    const size_t size = state.range(0);
    
    // 生成随机数据
    std::vector<float> f32_input = generateRandomDataF32(size);
    std::vector<float16_t> f16_input = convertF32ToF16(f32_input);
    std::vector<float16_t> f16_output(size);
    
    for (auto _ : state) {
        // 使用SLEEF计算
        for (size_t i = 0; i < size; i += 4) {
            #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            // 转换为float32
            float16x4_t f16x4 = vld1_f16((float16_t*)&f16_input[i]);
            float32x4_t f32x4 = vcvt_f32_f16(f16x4);
            
            // 使用SLEEF的NEON优化版本
            float32x4_t result_f32 = Sleef_expf4_u10advsimd(f32x4);
            
            // 转回float16
            float16x4_t result_f16 = vcvt_f16_f32(result_f32);
            vst1_f16((float16_t*)&f16_output[i], result_f16);
            #else
            // 手动转换，使用SLEEF标量函数
            float tmp_in[4], tmp_out[4];
            
            // 转换为float32
            for (int j = 0; j < 4 && (i + j) < size; ++j) {
                tmp_in[j] = (float)f16_input[i + j];
            }
            
            // 使用SLEEF的标量函数
            for (int j = 0; j < 4 && (i + j) < size; ++j) {
                tmp_out[j] = Sleef_expf1_u10(tmp_in[j]);
            }
            
            // 转回float16
            for (int j = 0; j < 4 && (i + j) < size; ++j) {
                f16_output[i + j] = (float16_t)tmp_out[j];
            }
            #endif
        }
        
        benchmark::DoNotOptimize(f16_output);
        benchmark::ClobberMemory();
    }
    
    // 设置性能指标
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float16_t));
    state.SetLabel("SLEEF FP16 via FP32");
}

// SLEEF的直接float32性能测试（作为参考）
static void BM_SLEEF_ExpF32_NEON(benchmark::State& state) {
    const size_t size = state.range(0);
    
    // 生成随机数据
    std::vector<float> f32_input = generateRandomDataF32(size);
    std::vector<float> f32_output(size);
    
    for (auto _ : state) {
        // 使用SLEEF计算
        for (size_t i = 0; i < size; i += 4) {
            float32x4_t input = vld1q_f32(&f32_input[i]);
            float32x4_t result = Sleef_expf4_u10advsimd(input);
            vst1q_f32(&f32_output[i], result);
        }
        
        benchmark::DoNotOptimize(f32_output);
        benchmark::ClobberMemory();
    }
    
    // 设置性能指标
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
    state.SetLabel("SLEEF FP32 NEON");
}

// 标准库的float32 exp性能测试（作为参考）
static void BM_Standard_ExpF32(benchmark::State& state) {
    const size_t size = state.range(0);
    
    // 生成随机数据
    std::vector<float> f32_input = generateRandomDataF32(size);
    std::vector<float> f32_output(size);
    
    for (auto _ : state) {
        // 使用标准库计算
        for (size_t i = 0; i < size; ++i) {
            f32_output[i] = expf(f32_input[i]);
        }
        
        benchmark::DoNotOptimize(f32_output);
        benchmark::ClobberMemory();
    }
    
    // 设置性能指标
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(float));
    state.SetLabel("Standard expf");
}

// 注册测试
BENCHMARK(BM_Custom_ExpF16_NEON)
    ->Arg(32 * 1024)        // 32K elements
    ->Arg(64 * 1024)        // 64K elements
    ->Arg(128 * 1024)       // 128K elements
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SLEEF_ExpF16_via_F32)
    ->Arg(32 * 1024)        // 32K elements
    ->Arg(64 * 1024)        // 64K elements
    ->Arg(128 * 1024)       // 128K elements
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SLEEF_ExpF32_NEON)
    ->Arg(32 * 1024)        // 32K elements
    ->Arg(64 * 1024)        // 64K elements
    ->Arg(128 * 1024)       // 128K elements
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Standard_ExpF32)
    ->Arg(32 * 1024)        // 32K elements
    ->Arg(64 * 1024)        // 64K elements
    ->Arg(128 * 1024)       // 128K elements
    ->Unit(benchmark::kMillisecond);

// 主函数
BENCHMARK_MAIN(); 