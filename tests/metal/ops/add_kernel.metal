#include <metal_stdlib>
using namespace metal;
kernel void add_kernel(const device float* a [[buffer(0)]],
                      const device float* b [[buffer(1)]],
                      device float* c [[buffer(2)]],
                      constant int& size [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        c[gid] = a[gid] + b[gid];
}
