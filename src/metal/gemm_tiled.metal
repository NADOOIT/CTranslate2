#include <metal_stdlib>
using namespace metal;

#define TILE 16

kernel void gemm_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    threadgroup float A_tile[TILE][TILE];
    threadgroup float B_tile[TILE][TILE];
    float acc = 0.0f;
    int row = gid.y;
    int col = gid.x;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int tiledRow = row;
        int tiledCol = t * TILE + tid.x;
        if (tiledRow < M && tiledCol < K)
            A_tile[tid.y][tid.x] = A[tiledRow * K + tiledCol];
        else
            A_tile[tid.y][tid.x] = 0.0f;

        tiledRow = t * TILE + tid.y;
        tiledCol = col;
        if (tiledRow < K && tiledCol < N)
            B_tile[tid.y][tid.x] = B[tiledRow * N + tiledCol];
        else
            B_tile[tid.y][tid.x] = 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int k = 0; k < TILE; ++k)
            acc += A_tile[tid.y][k] * B_tile[k][tid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}
