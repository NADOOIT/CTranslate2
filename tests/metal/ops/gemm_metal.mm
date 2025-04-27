#include "gemm_metal.h"
#include <iostream>
#include <chrono>

bool gemm_metal(
    id<MTLDevice> device,
    const std::vector<std::vector<float>>& a,
    const std::vector<std::vector<float>>& b,
    std::vector<std::vector<float>>& c,
    int m, int n, int k, int batch,
    int repeats,
    double& total_ms,
    std::string& diag_log,
    const std::string& mode)
{
    total_ms = 0.0;
    diag_log.clear();
    NSError* err = nil;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        diag_log += "[DIAG] Metal: failed to create command queue\n";
        return false;
    }
    if ((mode == "metal" && batch == 1) || (mode == "metal_batched" && batch > 1)) {
        id<MTLLibrary> lib = [device newDefaultLibrary];
        if (!lib) {
            diag_log += "[DIAG] Metal: missing MTLLibrary\n";
            return false;
        }
        id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
        if (!func) {
            diag_log += "[DIAG] Metal: missing gemm_kernel function\n";
            return false;
        }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
        if (!pipeline) {
            diag_log += "[DIAG] Metal: failed to create pipeline\n";
            return false;
        }
        for (int r = 0; r < repeats; ++r) {
            std::vector<id<MTLBuffer>> a_bufs(batch);
            std::vector<id<MTLBuffer>> b_bufs(batch);
            std::vector<id<MTLBuffer>> c_bufs(batch);
            for (int bidx = 0; bidx < batch; ++bidx) {
                a_bufs[bidx] = [device newBufferWithBytes:a[bidx].data() length:sizeof(float)*m*k options:MTLResourceStorageModePrivate];
                b_bufs[bidx] = [device newBufferWithBytes:b[bidx].data() length:sizeof(float)*k*n options:MTLResourceStorageModePrivate];
                c_bufs[bidx] = [device newBufferWithLength:sizeof(float)*m*n options:MTLResourceStorageModePrivate];
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
    if (mode == "metal_tiled") {
        diag_log += "[DIAG] Metal: mode 'metal_tiled' not implemented\n";
        return false;
    }
    if (mode == "metal_fp16" && batch == 1) {
        diag_log += "[DIAG] Metal: mode 'metal_fp16' not implemented\n";
        return false;
    }
    diag_log += "[DIAG] Metal: mode not implemented\n";
    return false;
}
