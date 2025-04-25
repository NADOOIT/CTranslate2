#include <metal_stdlib>
using namespace metal;

kernel void gemm_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant int* dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    int m = dims[0];
    int n = dims[1];
    int k = dims[2];
    int row = gid.x;
    int col = gid.y;
    if (row >= m || col >= n) return;
    float sum = 0.0f;
    for (int l = 0; l < k; ++l) {
        sum += a[row * k + l] * b[l * n + col];
    }
    c[row * n + col] = sum;
}
