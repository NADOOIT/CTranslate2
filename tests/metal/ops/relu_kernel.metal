#include <metal_stdlib>
using namespace metal;
kernel void relu_kernel(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size)
        out[gid] = in[gid] > 0.0f ? in[gid] : 0.0f;
}
