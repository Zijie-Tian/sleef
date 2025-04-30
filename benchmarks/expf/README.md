# SLEEF expf 与标准库 expf 性能比较基准测试（ARM NEON优化）

这个基准测试比较了不同版本指数函数实现的性能：
1. 标准 C 库的 `expf` 函数（标量版本）
2. SLEEF 库的 `Sleef_expf1_u10` 函数（标量版本）
3. 标准 C 库的 `expf` 函数（手动 NEON 向量化版本）
4. SLEEF 库的 `Sleef_expf4_u10advsimd` 函数（ARM NEON 优化版本）

重点是比较 SLEEF 的 ARM NEON 优化版本（使用 SIMD 指令并行处理 4 个浮点数）与其他实现的性能差异。

## 测试内容

- 对 32K、64K 和 128K 个随机浮点数进行 expf 计算
- 使用 Google Benchmark 库进行精确的性能计时
- 比较各种实现在大规模数据处理时的端到端延迟

## 先决条件

- 需要安装 Google Benchmark 库
- 必须在 ARM 架构（支持 NEON 指令集）上运行
- SLEEF 库必须已经构建并安装在项目根目录，且启用了 ARM NEON 支持

## 使用 CMake 构建（推荐）

```bash
mkdir build
cd build
cmake ..
make
./expf_benchmark
```

## 使用 Makefile 直接构建

```bash
make
./expf_benchmark
```

## 结果解读

SLEEF 的 ARM NEON 优化版本（`Sleef_expf4_u10advsimd`）应该在处理大量数据时比标量版本有显著的性能优势，因为它能够同时处理 4 个浮点数。标准库的手动向量化版本可能也会有所改进，但由于缺乏内置的 NEON exp 函数，它必须通过中间数组进行额外的内存操作。

## 注意事项

- 确保编译时启用了 ARM NEON 指令集支持（-march=armv8-a+simd）
- 此基准测试专为 ARM 架构设计，在 x86 等其他架构上无法运行
- 为获得最佳性能，测试使用 `-O3` 编译选项
- 确保在真实工作负载的代表性环境中运行基准测试，避免 CPU 频率调整和其他干扰 