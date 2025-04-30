#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
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
    float tmp_n[4], tmp_scale[4], tmp_result[4];
    vst1q_f32(tmp_n, n_f32);  // 将n存储到临时数组
    
    for (int i = 0; i < 4; i++) {
        tmp_scale[i] = std::exp2(tmp_n[i]);  // 使用标准库的exp2计算2^n
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

// 转换float16到float32，用于打印输出
std::vector<float> convertF16ToF32(const std::vector<float16_t>& f16_data) {
    std::vector<float> f32_data(f16_data.size());
    
    for (size_t i = 0; i < f16_data.size(); i += 4) {
        #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        float16x4_t f16x4 = vld1_f16((float16_t*)&f16_data[i]);
        float32x4_t f32x4 = vcvt_f32_f16(f16x4);
        vst1q_f32(&f32_data[i], f32x4);
        #else
        // 手动转换
        for (int j = 0; j < 4 && (i + j) < f16_data.size(); ++j) {
            f32_data[i + j] = (float)f16_data[i + j];
        }
        #endif
    }
    
    return f32_data;
}

// 生成测试数据
std::vector<float> generateTestData() {
    std::vector<float> data = {
        // 正常范围内的值
        0.0f, 0.5f, 1.0f, 2.0f,
        -0.5f, -1.0f, -2.0f, 3.0f,
        
        // 边界情况
        10.0f, 11.0f, 12.0f, -9.0f,
        -10.0f, -11.0f, -12.0f, 5.0f,
        
        // 特殊值
        INFINITY, -INFINITY, NAN, 7.0f
    };
    
    // 确保数据是4的倍数
    while (data.size() % 4 != 0) {
        data.push_back(0.0f);
    }
    
    return data;
}

// 计算相对误差
float relativeError(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) return 0.0f; // NAN与NAN比较为相等
    if (std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b)) return 0.0f; // 同号无穷大相等
    if (a == b) return 0.0f;
    if (std::abs(a) < 1e-30f || std::abs(b) < 1e-30f) 
        return std::abs(a - b); // 接近零时使用绝对误差
    return std::abs(a - b) / std::max(std::abs(a), std::abs(b));
}

// 打印比较结果
void printResults(const std::vector<float>& inputs, 
                 const std::vector<float>& custom_results,
                 const std::vector<float>& sleef_results) {
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=================================================================\n";
    std::cout << "   Input   |   Custom exp   |   SLEEF exp   |   Rel Error   |  Max ULP  \n";
    std::cout << "-----------------------------------------------------------------\n";
    
    float max_error = 0.0f;
    int total_tests = inputs.size();
    int passed_tests = 0;
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        float input = inputs[i];
        float custom_exp = custom_results[i];
        float sleef_exp = sleef_results[i];
        float error = relativeError(custom_exp, sleef_exp);
        
        // 估算ULP差距（非精确，仅作参考）
        int ulp = 0;
        if (!std::isnan(custom_exp) && !std::isnan(sleef_exp) && 
            !std::isinf(custom_exp) && !std::isinf(sleef_exp)) {
            ulp = std::abs(*(int*)&custom_exp - *(int*)&sleef_exp);
        }
        
        std::cout << std::setw(10) << input << " | ";
        std::cout << std::setw(14) << custom_exp << " | ";
        std::cout << std::setw(14) << sleef_exp << " | ";
        std::cout << std::setw(14) << error << " | ";
        std::cout << std::setw(9) << ulp;
        
        // 判断是否通过测试
        bool passed = (error < 1e-3f) || 
                    (std::isnan(custom_exp) && std::isnan(sleef_exp)) ||
                    (std::isinf(custom_exp) && std::isinf(sleef_exp) && 
                     std::signbit(custom_exp) == std::signbit(sleef_exp));
        
        std::cout << (passed ? "   ✓" : "   ✗") << std::endl;
        
        if (passed) passed_tests++;
        max_error = std::max(max_error, error);
    }
    
    std::cout << "=================================================================\n";
    std::cout << "测试结果: " << passed_tests << "/" << total_tests << " 通过\n";
    std::cout << "最大相对误差: " << max_error << std::endl;
}

//====================== 主函数 ======================

int main() {
    // 检查编译器对FP16和NEON的支持
    std::cout << "测试环境支持: " << std::endl;
    #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    std::cout << "FP16向量算术: 是" << std::endl;
    #else
    std::cout << "FP16向量算术: 否" << std::endl;
    #endif
    
    #ifdef __ARM_NEON
    std::cout << "ARM NEON: 是" << std::endl;
    #else
    std::cout << "ARM NEON: 否" << std::endl;
    #endif
    
    // 生成测试数据
    std::vector<float> test_data_f32 = generateTestData();
    std::vector<float16_t> test_data_f16 = convertF32ToF16(test_data_f32);
    
    // 结果向量
    std::vector<float16_t> custom_results(test_data_f16.size());
    std::vector<float16_t> sleef_results(test_data_f16.size());
    
    // 计算自定义实现的结果
    for (size_t i = 0; i < test_data_f16.size(); i += 4) {
        float16x4_t input = vld1_f16((float16_t*)&test_data_f16[i]);
        float16x4_t custom_result = expf_neon_f16x4(input);
        vst1_f16((float16_t*)&custom_results[i], custom_result);
    }
    
    // 计算SLEEF的结果 - 通过转换到float32，使用SLEEF，然后转回float16
    for (size_t i = 0; i < test_data_f16.size(); i += 4) {
        #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        // 转换为float32
        float16x4_t f16x4 = vld1_f16((float16_t*)&test_data_f16[i]);
        float32x4_t f32x4 = vcvt_f32_f16(f16x4);
        
        // 使用SLEEF的NEON优化版本
        float32x4_t result_f32 = Sleef_expf4_u10advsimd(f32x4);
        
        // 转回float16
        float16x4_t result_f16 = vcvt_f16_f32(result_f32);
        vst1_f16((float16_t*)&sleef_results[i], result_f16);
        #else
        // 手动转换，使用SLEEF标量函数
        float tmp_in[4], tmp_out[4];
        
        // 转换为float32
        for (int j = 0; j < 4 && (i + j) < test_data_f16.size(); ++j) {
            tmp_in[j] = (float)test_data_f16[i + j];
        }
        
        // 使用SLEEF的标量函数
        for (int j = 0; j < 4 && (i + j) < test_data_f16.size(); ++j) {
            tmp_out[j] = Sleef_expf1_u10(tmp_in[j]);
        }
        
        // 转回float16
        for (int j = 0; j < 4 && (i + j) < test_data_f16.size(); ++j) {
            sleef_results[i + j] = (float16_t)tmp_out[j];
        }
        #endif
    }
    
    // 转换结果为float32以便打印
    std::vector<float> custom_results_f32 = convertF16ToF32(custom_results);
    std::vector<float> sleef_results_f32 = convertF16ToF32(sleef_results);
    
    // 打印比较结果
    printResults(test_data_f32, custom_results_f32, sleef_results_f32);
    
    return 0;
} 