#include <metal_stdlib>
using namespace metal;

#define TILE 16

kernel void gemm_fp16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    threadgroup half A_tile[TILE][TILE];
    threadgroup half B_tile[TILE][TILE];
    half acc = 0.0h;
    int row = gid.y;
    int col = gid.x;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int tiledRow = row;
        int tiledCol = t * TILE + tid.x;
        if (tiledRow < M && tiledCol < K)
            A_tile[tid.y][tid.x] = A[tiledRow * K + tiledCol];
        else
            A_tile[tid.y][tid.x] = 0.0h;

        tiledRow = t * TILE + tid.y;
        tiledCol = col;
        if (tiledRow < K && tiledCol < N)
            B_tile[tid.y][tid.x] = B[tiledRow * N + tiledCol];
        else
            B_tile[tid.y][tid.x] = 0.0h;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int k = 0; k < TILE; ++k)
            acc += A_tile[tid.y][k] * B_tile[k][tid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}
