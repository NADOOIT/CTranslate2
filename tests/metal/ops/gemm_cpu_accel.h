#pragma once
#include <vector>
#include <string>

// CPU naive GEMM interface
void gemm_cpu(const float* a, const float* b, float* c, int m, int n, int k);

// Accelerate (cblas) GEMM interface
void gemm_accelerate(const float* a, const float* b, float* c, int m, int n, int k);

// GCD parallel GEMM (naive, each row)
void gemm_gcd(const float* a, const float* b, float* c, int m, int n, int k);

// Accelerate+GCD batched GEMM
void gemm_accelerate_gcd(const std::vector<std::vector<float>>& a,
                         const std::vector<std::vector<float>>& b,
                         std::vector<std::vector<float>>& c,
                         int m, int n, int k, int batch);
