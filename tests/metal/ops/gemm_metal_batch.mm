#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <iostream>
extern "C" bool gemm_metal_batch(const std::vector<const float*>& a, const std::vector<const float*>& b, std::vector<float*>& c, int m, int n, int k) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "[GEMM-Metal-Batch] Kein Metal-Device gefunden!" << std::endl;
            return false;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
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
                std::cerr << "[GEMM-Metal-Batch] Kein Metal-Library!" << std::endl;
                return false;
            }
        }
        id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
        if (!func) {
            std::cerr << "[GEMM-Metal-Batch] Kein gemm_kernel!" << std::endl;
            return false;
        }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
        if (!pipeline) {
            std::cerr << "[GEMM-Metal-Batch] Pipeline-Fehler!" << std::endl;
            return false;
        }
        for (size_t bidx = 0; bidx < a.size(); ++bidx) {
            id<MTLBuffer> a_buf = [device newBufferWithBytes:a[bidx] length:sizeof(float)*m*k options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_buf = [device newBufferWithBytes:b[bidx] length:sizeof(float)*k*n options:MTLResourceStorageModeShared];
            id<MTLBuffer> c_buf = [device newBufferWithLength:sizeof(float)*m*n options:MTLResourceStorageModeShared];
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
            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];
            memcpy(c[bidx], [c_buf contents], sizeof(float)*m*n);
        }
        return true;
    }
}
