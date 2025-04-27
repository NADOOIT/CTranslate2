#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
typedef __fp16 half;
#include <iostream>
#include <fstream>
#include <ctime>
#include <vector>
#include <random>
#include "gemm_metal.h"
#include "gemm_mps.h"
#include "gemm_cpu_accel.h"
#include "gemm_utils.h"
#include "gemm_bench_helpers.h"
#include <tuple>
#include <chrono>
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>

// Referenz-GEMM (CPU)
void gemm_cpu(const float* a, const float* b, float* c, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// Accelerate GEMM (single-threaded)
void gemm_accelerate(const float* a, const float* b, float* c, int m, int n, int k) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
}

// GCD parallel GEMM (naive, each row)
void gemm_gcd(const float* a, const float* b, float* c, int m, int n, int k) {
    dispatch_apply(m, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    });
}

int main() {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices;
        if (@available(macOS 10.13, *)) {
            devices = MTLCopyAllDevices();
        } else {
            devices = @[MTLCreateSystemDefaultDevice()];
        }
        const char* commit = "f35fc96e";
        const char* operator_name = "GEMM";
        const char* csv_file = "benchmarks_ops.csv";
        std::vector<std::tuple<int,int,int>> sizes = { {32,32,32}, {128,128,128}, {512,512,512} };
        std::vector<int> batch_sizes = {1, 8, 32};
        std::vector<std::string> modes = {"cpu_naive", "cpu_accelerate", "cpu_gcd", "cpu_accgcd", "metal", "metal_batched", "metal_tiled", "mps", "metal_fp16", "mps_fp16"};
        bool exists = file_exists(csv_file);
        std::ofstream csv(csv_file, std::ios::app);
        if (!exists) {
            write_csv_header(csv);
        }
        for (id<MTLDevice> device in devices) {
            NSString* device_name = [device name];
            bool isNeuralEngine = false;
            if ([device_name containsString:@"ANE"] || [device_name containsString:@"Neural"]) {
                isNeuralEngine = true;
            }
            std::cout << "[GEMM-BENCH] Device: " << [device_name UTF8String] << std::endl;
std::cerr << "[DIAG] Device Properties: " << [device_name UTF8String]
          << ", maxThreadsPerThreadgroup=" << device.maxThreadsPerThreadgroup.width << "x" << device.maxThreadsPerThreadgroup.height << "x" << device.maxThreadsPerThreadgroup.depth
          << ", recommendedMaxWorkingSetSize=" << device.recommendedMaxWorkingSetSize
          << ", supports MTLFeatureSet_macOS_GPUFamily1_v3=" << ([device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily1_v3] ? "YES" : "NO")
          << std::endl;
            for (const auto& sz : sizes) {
                int m = std::get<0>(sz), n = std::get<1>(sz), k = std::get<2>(sz);
                for (int batch : batch_sizes) {
                    for (const auto& mode : modes) {
    double avg_ms = 0.0;
    bool skipped = false;
                        std::vector<std::vector<float>> a(batch, std::vector<float>(m * k));
                        std::vector<std::vector<float>> b(batch, std::vector<float>(k * n));
                        std::vector<std::vector<float>> c(batch, std::vector<float>(m * n));
                        for (int bidx = 0; bidx < batch; ++bidx) {
                            fill_random(a[bidx], 42 + bidx);
                            fill_random(b[bidx], 142 + bidx);
                        }
                        double total_ms = 0.0;
                        const int repeats = 10;
                        if (mode == "cpu_naive" && batch == 1) {
                            std::cerr << "[DIAG] cpu_naive: running on CPU for device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl; 
                            for (int r = 0; r < repeats; ++r) {
                                auto start = std::chrono::high_resolution_clock::now();
                                gemm_cpu(a[0].data(), b[0].data(), c[0].data(), m, n, k);
                                auto end = std::chrono::high_resolution_clock::now();
                                total_ms += std::chrono::duration<double, std::milli>(end - start).count();
                            }
                        } else if (mode == "cpu_accelerate" && batch == 1) {
                            std::cerr << "[DIAG] cpu_accelerate: running on Accelerate for device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl; 
                            for (int r = 0; r < repeats; ++r) {
                                auto start = std::chrono::high_resolution_clock::now();
                                gemm_accelerate(a[0].data(), b[0].data(), c[0].data(), m, n, k);
                                auto end = std::chrono::high_resolution_clock::now();
                                total_ms += std::chrono::duration<double, std::milli>(end - start).count();
                            }
                        } else if (mode == "cpu_gcd") {
                            std::cerr << "[DIAG] cpu_gcd: running on CPU (GCD) for device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl; 
                            for (int r = 0; r < repeats; ++r) {
                                auto start = std::chrono::high_resolution_clock::now();
                                for (int bidx = 0; bidx < batch; ++bidx) {
                                    gemm_gcd(a[bidx].data(), b[bidx].data(), c[bidx].data(), m, n, k);
                                }
                                auto end = std::chrono::high_resolution_clock::now();
                                total_ms += std::chrono::duration<double, std::milli>(end - start).count();
                            }
                        } else if (mode == "cpu_accgcd") {
                            std::cerr << "[DIAG] cpu_accgcd: running on Accelerate (GCD) for device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl; 
                            for (int r = 0; r < repeats; ++r) {
                                auto start = std::chrono::high_resolution_clock::now();
                                gemm_accelerate_gcd(a, b, c, m, n, k, batch);
                                auto end = std::chrono::high_resolution_clock::now();
                                total_ms += std::chrono::duration<double, std::milli>(end - start).count();
                            }
                        } else if ((mode == "metal" && batch == 1) || (mode == "metal_batched" && batch > 1) || mode == "metal_tiled" || (mode == "metal_fp16" && batch == 1)) {
                            std::string diag_log;
                            bool metal_ok = gemm_metal(device, a, b, c, m, n, k, batch, repeats, total_ms, diag_log, mode);
                            if (!metal_ok) skipped = true;
                            if (!diag_log.empty()) std::cerr << diag_log;
                        } else if (mode == "mps" && batch == 1) {
                            std::string diag_log;
                            bool mps_ok = gemm_mps(device, a, b, c, m, n, k, batch, repeats, total_ms, diag_log);
                            if (!mps_ok) skipped = true;
                            if (!diag_log.empty()) std::cerr << diag_log;
                        } else if (mode == "mps_fp16" && batch == 1) {
                            std::string diag_log;
                            bool mps16_ok = gemm_mps_fp16(device, a, b, c, m, n, k, batch, repeats, total_ms, diag_log);
                            if (!mps16_ok) skipped = true;
                            if (!diag_log.empty()) std::cerr << diag_log;
                        } else {
                            std::cerr << "[DIAG] skipped: mode '" << mode << "' is not handled for device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl;
                            skipped = true;
                        }
                        if (total_ms > 0.0) {
                            avg_ms = total_ms / repeats;
                            char size_str[32];
                            snprintf(size_str, sizeof(size_str), "%dx%dx%d", m, n, k);
                            std::time_t t = std::time(nullptr);
                            char timebuf[32];
                            std::strftime(timebuf, sizeof(timebuf), "%FT%T%z", std::localtime(&t));
                            csv << timebuf << "," << commit << ",GEMM," << mode << "," << [device_name UTF8String] << "," << size_str << "," << batch << "," << avg_ms << "," << (skipped ? "skipped" : "ok") << "\n";
                            std::cout << "[BENCHMARK][" << [device_name UTF8String] << "] " << mode << " size=" << size_str << " batch=" << batch << ": " << (skipped ? "skipped" : std::to_string(avg_ms) + " ms") << std::endl;
                        }

                            id<MTLCommandQueue> queue = [device newCommandQueue];
                            NSError* err = nil;
                            id<MTLLibrary> lib = [device newDefaultLibrary];
                            if (!lib) { std::cerr << "[DIAG] metal_batched: missing MTLLibrary for device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl; continue; }
                            id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
                            if (!func) { std::cerr << "[DIAG] metal_batched: missing gemm_kernel function for device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl; continue; }
                            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                            if (!pipeline) { std::cerr << "[DIAG] metal_batched: failed to create pipeline for device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl; continue; }
                            for (int r = 0; r < repeats; ++r) {
                                std::vector<id<MTLBuffer>> a_bufs(batch);
                                std::vector<id<MTLBuffer>> b_bufs(batch);
                                std::vector<id<MTLBuffer>> c_bufs(batch);
                                for (int bidx = 0; bidx < batch; ++bidx) {
                                    std::cerr << "[DIAG] metal_batched: allocating a_buf[" << bidx << "] for device " << [device_name UTF8String] << ", size=" << m << "x" << k << ", batch=" << batch << std::endl;
                                    a_bufs[bidx] = [device newBufferWithBytes:a[bidx].data() length:sizeof(float)*m*k options:MTLResourceStorageModePrivate];
                                    if (!a_bufs[bidx]) std::cerr << "[DIAG] metal_batched: a_buf[" << bidx << "] allocation FAILED for device " << [device_name UTF8String] << ", size=" << m << "x" << k << ", batch=" << batch << std::endl;
                                    std::cerr << "[DIAG] metal_batched: allocating b_buf[" << bidx << "] for device " << [device_name UTF8String] << ", size=" << k << "x" << n << ", batch=" << batch << std::endl;
                                    b_bufs[bidx] = [device newBufferWithBytes:b[bidx].data() length:sizeof(float)*k*n options:MTLResourceStorageModePrivate];
                                    if (!b_bufs[bidx]) std::cerr << "[DIAG] metal_batched: b_buf[" << bidx << "] allocation FAILED for device " << [device_name UTF8String] << ", size=" << k << "x" << n << ", batch=" << batch << std::endl;
                                    std::cerr << "[DIAG] metal_batched: allocating c_buf[" << bidx << "] for device " << [device_name UTF8String] << ", size=" << m << "x" << n << ", batch=" << batch << std::endl;
                                    c_bufs[bidx] = [device newBufferWithLength:sizeof(float)*m*n options:MTLResourceStorageModePrivate];
                                    if (!c_bufs[bidx]) std::cerr << "[DIAG] metal_batched: c_buf[" << bidx << "] allocation FAILED for device " << [device_name UTF8String] << ", size=" << m << "x" << n << ", batch=" << batch << std::endl;
                                }
                                std::vector<id<MTLCommandBuffer>> cmd_bufs{batch};
                                std::vector<id<MTLComputeCommandEncoder>> encoders{batch};
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
                        } else if (mode == "metal_tiled") {
                            // ... (existing metal_tiled code)
                        } else if (mode == "mps" && batch == 1) {
                            // ... (existing MPS code)
                        } else if (mode == "metal_fp16" && batch == 1) {
                            // ... (existing metal_fp16 code)
                        } else if (mode == "mps_fp16" && batch == 1) {
                            // Check fp16 support
                            if (![device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily1_v3]) { skipped = true; } else {
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
                            }
                        
                            // Check fp16 support
                            if (![device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily1_v3]) { skipped = true; } else {
                                // Convert float32 to half on CPU
                                std::vector<half> a16(m*k);
                                std::vector<half> b16(k*n);
                                for (int i = 0; i < m*k; ++i) a16[i] = static_cast<half>(a[0][i]);
                                for (int i = 0; i < k*n; ++i) b16[i] = static_cast<half>(b[0][i]);
                                id<MTLCommandQueue> queue = [device newCommandQueue];
                                NSError* err = nil;
                                id<MTLLibrary> lib = [device newDefaultLibrary];
                                if (!lib) { std::cerr << "[DIAG] metal_fp16: missing MTLLibrary for device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl; skipped = true; } else {
                                    id<MTLFunction> func = [lib newFunctionWithName:@"gemm_fp16"];
                                    if (!func) { std::cerr << "[DIAG] metal_fp16: missing gemm_fp16 function for device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl; skipped = true; } else {
                                        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                                        if (!pipeline) { std::cerr << "[DIAG] metal_fp16: failed to create pipeline for device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl; skipped = true; } else {
                                            std::cerr << "[DIAG] metal_fp16: allocating a_buf for device " << [device_name UTF8String] << ", size=" << m << "x" << k << ", batch=" << batch << std::endl;
                                            id<MTLBuffer> a_buf = [device newBufferWithBytes:a16.data() length:sizeof(half)*m*k options:MTLResourceStorageModePrivate];
                                            if (!a_buf) std::cerr << "[DIAG] metal_fp16: a_buf allocation FAILED for device " << [device_name UTF8String] << ", size=" << m << "x" << k << ", batch=" << batch << std::endl;
                                            std::cerr << "[DIAG] metal_fp16: allocating b_buf for device " << [device_name UTF8String] << ", size=" << k << "x" << n << ", batch=" << batch << std::endl;
                                            id<MTLBuffer> b_buf = [device newBufferWithBytes:b16.data() length:sizeof(half)*k*n options:MTLResourceStorageModePrivate];
                                            if (!b_buf) std::cerr << "[DIAG] metal_fp16: b_buf allocation FAILED for device " << [device_name UTF8String] << ", size=" << k << "x" << n << ", batch=" << batch << std::endl;
                                            std::cerr << "[DIAG] metal_fp16: allocating c_buf for device " << [device_name UTF8String] << ", size=" << m << "x" << n << ", batch=" << batch << std::endl;
                                            id<MTLBuffer> c_buf = [device newBufferWithLength:sizeof(half)*m*n options:MTLResourceStorageModePrivate];
                                            if (!c_buf) std::cerr << "[DIAG] metal_fp16: c_buf allocation FAILED for device " << [device_name UTF8String] << ", size=" << m << "x" << n << ", batch=" << batch << std::endl;
                                            for (int r = 0; r < repeats; ++r) {
                                                std::cerr << "[DIAG] metal_fp16: kernel dispatch starting for device " << [device_name UTF8String] << ", mode=metal_fp16, size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl;
                                                id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
                                                id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
                                                [encoder setComputePipelineState:pipeline];
                                                [encoder setBuffer:a_buf offset:0 atIndex:0];
                                                [encoder setBuffer:b_buf offset:0 atIndex:1];
                                                [encoder setBuffer:c_buf offset:0 atIndex:2];
                                                [encoder setBytes:&m length:sizeof(int) atIndex:3];
                                                [encoder setBytes:&n length:sizeof(int) atIndex:4];
                                                [encoder setBytes:&k length:sizeof(int) atIndex:5];
                                                MTLSize grid = MTLSizeMake(n, m, 1);
                                                MTLSize threadgroup = MTLSizeMake(16, 16, 1);
                                                [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
                                                [encoder endEncoding];
                                                std::cerr << "[DIAG] metal_fp16: committing command buffer for device " << [device_name UTF8String] << ", mode=metal_fp16, size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl;
                                                [cmd_buf commit];
                                                [cmd_buf waitUntilCompleted];
                                                std::cerr << "[DIAG] metal_fp16: command buffer completed for device " << [device_name UTF8String] << ", mode=metal_fp16, size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl;
                                                auto start = std::chrono::high_resolution_clock::now();
                                                [cmd_buf commit];
                                                [cmd_buf waitUntilCompleted];
                                                auto end = std::chrono::high_resolution_clock::now();
                                                total_ms += std::chrono::duration<double, std::milli>(end - start).count();
                                            }
                                        }
                                    }
                                }
                            }
                        
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
                            } @catch (NSException *exception) {
                                std::cerr << "[DIAG] mps: exception caught on device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << ": " << [[exception reason] UTF8String] << std::endl;
                                skipped = true;
                            }
                        
                            if (batch != 1) { std::cerr << "[DIAG] metal_tiled: batch != 1, skipping for device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl; skipped = true; } else {
                                id<MTLCommandQueue> queue = [device newCommandQueue];
                                NSError* err = nil;
                                id<MTLLibrary> lib = [device newDefaultLibrary];
                                                [encoder setBuffer:c_buf offset:0 atIndex:2];
                                                [encoder setBytes:&m length:sizeof(int) atIndex:3];
                                                [encoder setBytes:&n length:sizeof(int) atIndex:4];
                                                [encoder setBytes:&k length:sizeof(int) atIndex:5];
                                                MTLSize grid = MTLSizeMake(n, m, 1);
                                                MTLSize threadgroup = MTLSizeMake(16, 16, 1);
                                                [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
                                                [encoder endEncoding];
                                                auto start = std::chrono::high_resolution_clock::now();
                                                [cmd_buf commit];
                                                [cmd_buf waitUntilCompleted];
                                                auto end = std::chrono::high_resolution_clock::now();
                                                total_ms += std::chrono::duration<double, std::milli>(end - start).count();
                                            }
                                        }
                                    }
                                }
                            }
                            id<MTLCommandQueue> queue = [device newCommandQueue];
                            NSError* err = nil;
                            id<MTLLibrary> lib = [device newDefaultLibrary];
                            if (!lib) { skipped = true; }
                            id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
                            if (!func) { skipped = true; }
                             id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                             if (!pipeline) { skipped = true; }
                         }
                         else {
                             std::cerr << "[DIAG] skipped: mode '" << mode << "' is not handled for device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl;
                             skipped = true;
                         }
                         if (total_ms > 0.0) {
                            avg_ms = total_ms / repeats;
                            char size_str[32];
                            snprintf(size_str, sizeof(size_str), "%dx%dx%d", m, n, k);
                            std::time_t t = std::time(nullptr);
                            char timebuf[32];
                            std::strftime(timebuf, sizeof(timebuf), "%FT%T%z", std::localtime(&t));
                            csv << timebuf << "," << commit << ",GEMM," << mode << "," << [device_name UTF8String] << "," << size_str << "," << batch << "," << avg_ms << "," << (skipped ? "skipped" : "ok") << "\n";
                            std::cout << "[BENCHMARK][" << [device_name UTF8String] << "] " << mode << " size=" << size_str << " batch=" << batch << ": " << (skipped ? "skipped" : std::to_string(avg_ms) + " ms") << std::endl;
                        }

            for (const auto& sz : sizes) {
                int m = std::get<0>(sz), n = std::get<1>(sz), k = std::get<2>(sz);
                std::vector<float> a(m * k), b(k * n), c_ref(m * n), c_metal(m * n);
                std::mt19937 gen(42);
                std::uniform_real_distribution<float> dist(-1, 1);
                for (auto& v : a) v = dist(gen);
                for (auto& v : b) v = dist(gen);
                // --- CPU (Naive) ---
                const int repeats = 10;
                double cpu_total = 0.0;
                for (int r = 0; r < repeats; ++r) {
                    auto start = std::chrono::high_resolution_clock::now();
                    gemm_cpu(a.data(), b.data(), c_ref.data(), m, n, k);
                    auto end = std::chrono::high_resolution_clock::now();
                    cpu_total += std::chrono::duration<double, std::milli>(end - start).count();
                }
                double cpu_ms = cpu_total / repeats;
                std::cout << "[BENCHMARK] CPU (naive): " << cpu_ms << " ms" << std::endl;

                // --- CPU (Accelerate) ---
                double acc_total = 0.0;
                std::vector<float> c_acc(m * n);
                for (int r = 0; r < repeats; ++r) {
                    auto start = std::chrono::high_resolution_clock::now();
                    gemm_accelerate(a.data(), b.data(), c_acc.data(), m, n, k);
                    auto end = std::chrono::high_resolution_clock::now();
                    acc_total += std::chrono::duration<double, std::milli>(end - start).count();
                }
                double acc_ms = acc_total / repeats;
                std::cout << "[BENCHMARK] CPU (Accelerate): " << acc_ms << " ms" << std::endl;

                // --- CPU (GCD naive) ---
                double gcd_total = 0.0;
                std::vector<float> c_gcd(m * n);
                for (int r = 0; r < repeats; ++r) {
                    auto start = std::chrono::high_resolution_clock::now();
                    gemm_gcd(a.data(), b.data(), c_gcd.data(), m, n, k);
                    auto end = std::chrono::high_resolution_clock::now();
                    gcd_total += std::chrono::duration<double, std::milli>(end - start).count();
                }
                double gcd_ms = gcd_total / repeats;
                std::cout << "[BENCHMARK] CPU (GCD naive): " << gcd_ms << " ms" << std::endl;

                // --- Batched GEMM Benchmarks (all CPU modes) ---
                run_cpu_batched_benchmarks(sizes, batch_sizes, repeats, csv, commit, operator_name, device, [device_name UTF8String]);

                // --- Metal (single device, single batch) ---
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
                        std::cerr << "[GEMM-BENCH] Kein Metal-Library auf Device: " << [device_name UTF8String] << std::endl;
                        continue;
                    }
                }
                id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
                if (!func) {
                    std::cerr << "[GEMM-BENCH] Kein gemm_kernel auf Device: " << [device_name UTF8String] << std::endl;
                    continue;
                }
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                if (!pipeline) {
                    std::cerr << "[DIAG] metal: failed to create pipeline for device " << [device_name UTF8String] << ", size=" << m << "x" << n << "x" << k << ", batch=" << batch << std::endl; 
                    continue;
                }
                id<MTLCommandQueue> queue = [device newCommandQueue];
                id<MTLBuffer> a_buf = [device newBufferWithBytes:a.data() length:sizeof(float)*a.size() options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_buf = [device newBufferWithBytes:b.data() length:sizeof(float)*b.size() options:MTLResourceStorageModeShared];
                id<MTLBuffer> c_buf = [device newBufferWithLength:sizeof(float)*c_metal.size() options:MTLResourceStorageModeShared];
                if (!a_buf || !b_buf || !c_buf) {
                    std::cerr << "[GEMM-BENCH] Buffer-Fehler auf Device: " << [device_name UTF8String] << std::endl;
                    continue;
                }
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
                memcpy(c_metal.data(), [c_buf contents], sizeof(float)*c_metal.size());
                double max_diff_val = max_diff(c_metal.data(), c_ref.data(), c_metal.size());
                char size_str[32];
                snprintf(size_str, sizeof(size_str), "%dx%dx%d", m, n, k);
                std::string timebuf = current_time_string();
                if (max_diff < 1e-4) {
                    std::cout << "[BENCHMARK][" << [device_name UTF8String] << "] CPU: " << cpu_ms << " ms, Metal: " << metal_ms << " ms, Speedup: " << (cpu_ms / metal_ms) << std::endl;
                }

                // --- Metal Batched GEMM (all sizes/batches) ---
                run_metal_batched_benchmarks(sizes, batch_sizes, repeats, csv, commit, operator_name, device, [device_name UTF8String]);
                    csv << timebuf << "," << commit << "," << operator_name << "," << (isNeuralEngine ? "NeuralEngine" : "MetalGPU") << "," << size_str << "," << cpu_ms << "," << metal_ms << "," << (cpu_ms / metal_ms) << "\n";
                }
            }
        }
    }
    return 0;
}
