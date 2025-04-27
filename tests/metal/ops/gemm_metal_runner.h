#pragma once
#include <Metal/Metal.h>

struct MetalGEMMResult {
    double metal_ms;
    double max_diff;
};

MetalGEMMResult run_metal_gemm(id<MTLDevice> device, id<MTLCommandQueue> queue,
    const float* a, const float* b, float* c_metal, const float* c_ref,
    int m, int n, int k, int repeats);
