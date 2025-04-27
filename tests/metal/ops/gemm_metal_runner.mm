#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <tuple>
#include <iostream>
#include "gemm_metal_runner.h"

MetalGEMMResult run_metal_gemm(id<MTLDevice> device, id<MTLCommandQueue> queue, 
    const float* a, const float* b, float* c_metal, const float* c_ref, 
    int m, int n, int k, int repeats) {
    NSError* err = nil;
    id<MTLLibrary> lib = [device newDefaultLibrary];
    if (!lib) {
        NSString* path = @"gemm_kernel.metallib";
        NSFileManager* fm = [NSFileManager defaultManager];
        if (![fm fileExistsAtPath:path]) {
            NSString* exePath = [[NSBundle mainBundle] executablePath];
            NSString* exeDir = [exePath stringByDeletingLastPathComponent];
            path = [exeDir stringByAppendingPathComponent:@"gemm_kernel.metallib"];
        }
        NSError* libErr = nil;
        lib = [device newLibraryWithFile:path error:&libErr];
        if (!lib) {
            std::cerr << "[GEMMTest] Kein Metal-Library! (auch nicht als Datei)\n";
            if (libErr)
                std::cerr << "Fehler: " << [[libErr localizedDescription] UTF8String] << std::endl;
            throw std::runtime_error("Metal library not found");
        }
    }
    id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
    if (!func) throw std::runtime_error("gemm_kernel not found");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
    if (!pipeline) throw std::runtime_error("Pipeline error");

    id<MTLBuffer> a_buf = [device newBufferWithBytes:a length:sizeof(float)*m*k options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_buf = [device newBufferWithBytes:b length:sizeof(float)*k*n options:MTLResourceStorageModeShared];
    id<MTLBuffer> c_buf = [device newBufferWithLength:sizeof(float)*m*n options:MTLResourceStorageModeShared];
    if (!a_buf || !b_buf || !c_buf) throw std::runtime_error("Buffer error");

    double metal_total = 0.0;
    for (int r = 0; r < repeats; ++r) {
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a_buf offset:0 atIndex:0];
        [encoder setBuffer:b_buf offset:0 atIndex:1];
        [encoder setBuffer:c_buf offset:0 atIndex:2];
        int dims[3] = {m, n, k};
        [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
        MTLSize grid = MTLSizeMake(m, n, 1);
        MTLSize threadgroup = MTLSizeMake(8, 8, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];
        auto start = std::chrono::high_resolution_clock::now();
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
        auto end = std::chrono::high_resolution_clock::now();
        metal_total += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double metal_ms = metal_total / repeats;
    memcpy(c_metal, [c_buf contents], sizeof(float)*m*n);
    double max_diff = 0.0;
    for (size_t i = 0; i < m*n; ++i)
        max_diff = std::max(max_diff, static_cast<double>(std::abs(c_metal[i] - c_ref[i])));
    return {metal_ms, max_diff};
}
