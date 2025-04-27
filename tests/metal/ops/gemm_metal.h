#pragma once
#include <vector>
#include <string>
#include <Metal/Metal.h>

// Metal GEMM interface
bool gemm_metal(
    id<MTLDevice> device,
    const std::vector<std::vector<float>>& a,
    const std::vector<std::vector<float>>& b,
    std::vector<std::vector<float>>& c,
    int m, int n, int k, int batch,
    int repeats,
    double& total_ms,
    std::string& diag_log);
