#include "gemm_metal.h"
#include <vector>
#include <string>
#include <Metal/Metal.h>
#include <chrono>
typedef __fp16 half;

// Handles the 'metal_fp16' mode for Metal GEMM with FP16
bool gemm_metal_fp16(
    id<MTLDevice> device,
    const std::vector<std::vector<float>>& a,
    const std::vector<std::vector<float>>& b,
    std::vector<std::vector<float>>& c,
    int m, int n, int k, int batch,
    int repeats,
    double& total_ms,
    std::string& diag_log)
{
    total_ms = 0.0;
    diag_log.clear();
    NSError* err = nil;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        diag_log += "[DIAG] MetalFP16: failed to create command queue\n";
        return false;
    }
    id<MTLLibrary> lib = [device newDefaultLibrary];
    if (!lib) {
        diag_log += "[DIAG] MetalFP16: missing MTLLibrary\n";
        return false;
    }
    id<MTLFunction> func = [lib newFunctionWithName:@"gemm_fp16"];
    if (!func) {
        diag_log += "[DIAG] MetalFP16: missing gemm_fp16 function\n";
        return false;
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
    if (!pipeline) {
        diag_log += "[DIAG] MetalFP16: failed to create pipeline\n";
        return false;
    }
    for (int r = 0; r < repeats; ++r) {
        std::vector<id<MTLBuffer>> a_bufs(batch);
        std::vector<id<MTLBuffer>> b_bufs(batch);
        std::vector<id<MTLBuffer>> c_bufs(batch);
        for (int bidx = 0; bidx < batch; ++bidx) {
            // Convert float32 to half on CPU
            std::vector<half> a16(m*k);
            std::vector<half> b16(k*n);
            for (int i = 0; i < m*k; ++i) a16[i] = static_cast<half>(a[bidx][i]);
            for (int i = 0; i < k*n; ++i) b16[i] = static_cast<half>(b[bidx][i]);
            a_bufs[bidx] = [device newBufferWithBytes:a16.data() length:sizeof(half)*m*k options:MTLResourceStorageModePrivate];
            b_bufs[bidx] = [device newBufferWithBytes:b16.data() length:sizeof(half)*k*n options:MTLResourceStorageModePrivate];
            c_bufs[bidx] = [device newBufferWithLength:sizeof(half)*m*n options:MTLResourceStorageModePrivate];
        }
        std::vector<id<MTLCommandBuffer>> cmd_bufs(batch);
        std::vector<id<MTLComputeCommandEncoder>> encoders(batch);
        auto start = std::chrono::high_resolution_clock::now();
        for (int bidx = 0; bidx < batch; ++bidx) {
            cmd_bufs[bidx] = [queue commandBuffer];
            encoders[bidx] = [cmd_bufs[bidx] computeCommandEncoder];
            [encoders[bidx] setComputePipelineState:pipeline];
            [encoders[bidx] setBuffer:a_bufs[bidx] offset:0 atIndex:0];
            [encoders[bidx] setBuffer:b_bufs[bidx] offset:0 atIndex:1];
            [encoders[bidx] setBuffer:c_bufs[bidx] offset:0 atIndex:2];
            int dims[3] = {m, n, k};
            [encoders[bidx] setBytes:&dims length:sizeof(dims) atIndex:3];
            MTLSize grid = MTLSizeMake(m, n, 1);
            MTLSize threadgroup = MTLSizeMake(8, 8, 1);
            [encoders[bidx] dispatchThreads:grid threadsPerThreadgroup:threadgroup];
            [encoders[bidx] endEncoding];
            [cmd_bufs[bidx] commit];
        }
        for (int bidx = 0; bidx < batch; ++bidx) {
            [cmd_bufs[bidx] waitUntilCompleted];
        }
        auto end = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    return true;
}
