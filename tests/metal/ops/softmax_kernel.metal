#include <metal_stdlib>
using namespace metal;
kernel void softmax_kernel(const device float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant int& size [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    // Numerisch stabil: max abziehen
    float maxval = -FLT_MAX;
    for (uint i = 0; i < size; ++i)
        maxval = fmax(maxval, in[i]);
    float sum = 0.0f;
    for (uint i = 0; i < size; ++i)
        sum += exp(in[i] - maxval);
    if (gid < size)
        out[gid] = exp(in[gid] - maxval) / sum;
}
