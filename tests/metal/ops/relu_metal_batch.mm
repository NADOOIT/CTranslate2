#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <iostream>

extern "C" bool relu_metal_batch(const std::vector<const float*>& inputs, std::vector<float*>& outputs, int size) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "[ReLU-Metal-Batch] Kein Metal-Device gefunden!" << std::endl;
            return false;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [device newDefaultLibrary];
        if (!lib) {
            NSString* path = @"relu_kernel.metallib";
            NSFileManager* fm = [NSFileManager defaultManager];
            if (![fm fileExistsAtPath:path]) {
                NSString* exePath = [[NSBundle mainBundle] executablePath];
                NSString* exeDir = [exePath stringByDeletingLastPathComponent];
                path = [exeDir stringByAppendingPathComponent:@"relu_kernel.metallib"];
            }
            NSError* libErr = nil;
            lib = [device newLibraryWithFile:path error:&libErr];
            if (!lib) {
                std::cerr << "[ReLU-Metal-Batch] Kein Metal-Library!" << std::endl;
                return false;
            }
        }
        id<MTLFunction> func = [lib newFunctionWithName:@"relu_kernel"];
        if (!func) {
            std::cerr << "[ReLU-Metal-Batch] Kein relu_kernel!" << std::endl;
            return false;
        }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
        if (!pipeline) {
            std::cerr << "[ReLU-Metal-Batch] Pipeline-Fehler!" << std::endl;
            return false;
        }
        for (size_t b = 0; b < inputs.size(); ++b) {
            id<MTLBuffer> in_buf = [device newBufferWithBytes:inputs[b] length:sizeof(float)*size options:MTLResourceStorageModeShared];
            id<MTLBuffer> out_buf = [device newBufferWithLength:sizeof(float)*size options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:in_buf offset:0 atIndex:0];
            [encoder setBuffer:out_buf offset:0 atIndex:1];
            int sz = size;
            [encoder setBytes:&sz length:sizeof(int) atIndex:2];
            MTLSize grid = MTLSizeMake(size, 1, 1);
            MTLSize threadgroup = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
            [encoder endEncoding];
            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];
            memcpy(outputs[b], [out_buf contents], sizeof(float)*size);
        }
        return true;
    }
}
