#include "gemm_mps.h"
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>
typedef __fp16 half;

// Handles the 'mps' mode for MPS GEMM (float32)
bool gemm_mps(
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
    @try {
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> a_buf = [device newBufferWithBytes:a[0].data() length:sizeof(float)*m*k options:MTLResourceStorageModePrivate];
        id<MTLBuffer> b_buf = [device newBufferWithBytes:b[0].data() length:sizeof(float)*k*n options:MTLResourceStorageModePrivate];
        id<MTLBuffer> c_buf = [device newBufferWithLength:sizeof(float)*m*n options:MTLResourceStorageModePrivate];
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:n rowBytes:n*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:a_buf descriptor:descA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:b_buf descriptor:descB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:c_buf descriptor:descC];
        MPSMatrixMultiplication *mpsGemm = [[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:NO resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];
        for (int r = 0; r < repeats; ++r) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
            [mpsGemm encodeToCommandBuffer:cmd_buf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
            auto start = std::chrono::high_resolution_clock::now();
            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];
            auto end = std::chrono::high_resolution_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        return true;
    } @catch (NSException *exception) {
        diag_log += "[DIAG] mps: exception caught: ";
        diag_log += [[exception reason] UTF8String];
        diag_log += "\n";
        return false;
    }
}

// Handles the 'mps_fp16' mode for MPS GEMM (float16)
bool gemm_mps_fp16(
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
    if (![device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily1_v3]) {
        diag_log += "[DIAG] mps_fp16: device does not support required feature set\n";
        return false;
    }
    @try {
        // Convert float32 to half on CPU
        std::vector<half> a16(m*k);
        std::vector<half> b16(k*n);
        for (int i = 0; i < m*k; ++i) a16[i] = static_cast<half>(a[0][i]);
        for (int i = 0; i < k*n; ++i) b16[i] = static_cast<half>(b[0][i]);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> a_buf = [device newBufferWithBytes:a16.data() length:sizeof(half)*m*k options:MTLResourceStorageModePrivate];
        id<MTLBuffer> b_buf = [device newBufferWithBytes:b16.data() length:sizeof(half)*k*n options:MTLResourceStorageModePrivate];
        id<MTLBuffer> c_buf = [device newBufferWithLength:sizeof(half)*m*n options:MTLResourceStorageModePrivate];
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k*sizeof(half) dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:n rowBytes:n*sizeof(half) dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n*sizeof(half) dataType:MPSDataTypeFloat16];
        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:a_buf descriptor:descA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:b_buf descriptor:descB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:c_buf descriptor:descC];
        MPSMatrixMultiplication *mpsGemm = [[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:NO resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];
        for (int r = 0; r < repeats; ++r) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
            [mpsGemm encodeToCommandBuffer:cmd_buf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
            auto start = std::chrono::high_resolution_clock::now();
            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];
            auto end = std::chrono::high_resolution_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        return true;
    } @catch (NSException *exception) {
        diag_log += "[DIAG] mps_fp16: exception caught: ";
        diag_log += [[exception reason] UTF8String];
        diag_log += "\n";
        return false;
    }
}
