#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <vector>
#include <random>
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
        std::ifstream check(csv_file);
        bool file_exists = check.good();
        check.close();
        std::ofstream csv(csv_file, std::ios::app);
        if (!file_exists) {
            csv << "timestamp,commit,operator,mode,device,size,batch,avg_ms\n";
        }
        for (id<MTLDevice> device in devices) {
            NSString* device_name = [device name];
            bool isNeuralEngine = false;
            if ([device_name containsString:@"ANE"] || [device_name containsString:@"Neural"]) {
                isNeuralEngine = true;
            }
            std::cout << "[GEMM-BENCH] Device: " << [device_name UTF8String] << std::endl;
            for (const auto& sz : sizes) {
                int m = std::get<0>(sz), n = std::get<1>(sz), k = std::get<2>(sz);
                for (int batch : batch_sizes) {
                    for (const auto& mode : modes) {
                        std::vector<std::vector<float>> a(batch, std::vector<float>(m * k));
                        std::vector<std::vector<float>> b(batch, std::vector<float>(k * n));
                        std::vector<std::vector<float>> c(batch, std::vector<float>(m * n));
                        std::mt19937 gen(42);
                        std::uniform_real_distribution<float> dist(-1, 1);
                        for (int bidx = 0; bidx < batch; ++bidx) {
                            for (auto& v : a[bidx]) v = dist(gen);
                            for (auto& v : b[bidx]) v = dist(gen);
                        }
                        double total_ms = 0.0;
                        const int repeats = 10;
                        if (mode == "cpu_naive" && batch == 1) {
                            for (int r = 0; r < repeats; ++r) {
                                auto start = std::chrono::high_resolution_clock::now();
                                gemm_cpu(a[0].data(), b[0].data(), c[0].data(), m, n, k);
                                auto end = std::chrono::high_resolution_clock::now();
                                total_ms += std::chrono::duration<double, std::milli>(end - start).count();
                            }
                        } else if (mode == "cpu_accelerate" && batch == 1) {
                            for (int r = 0; r < repeats; ++r) {
                                auto start = std::chrono::high_resolution_clock::now();
                                gemm_accelerate(a[0].data(), b[0].data(), c[0].data(), m, n, k);
                                auto end = std::chrono::high_resolution_clock::now();
                                total_ms += std::chrono::duration<double, std::milli>(end - start).count();
                            }
                        } else if (mode == "cpu_gcd") {
                            for (int r = 0; r < repeats; ++r) {
                                auto start = std::chrono::high_resolution_clock::now();
                                dispatch_apply(batch, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t bidx) {
                                    gemm_gcd(a[bidx].data(), b[bidx].data(), c[bidx].data(), m, n, k);
                                });
                                auto end = std::chrono::high_resolution_clock::now();
                                total_ms += std::chrono::duration<double, std::milli>(end - start).count();
                            }
                        } else if (mode == "cpu_accgcd") {
                            for (int r = 0; r < repeats; ++r) {
                                auto start = std::chrono::high_resolution_clock::now();
                                dispatch_apply(batch, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t bidx) {
                                    gemm_accelerate(a[bidx].data(), b[bidx].data(), c[bidx].data(), m, n, k);
                                });
                                auto end = std::chrono::high_resolution_clock::now();
                                total_ms += std::chrono::duration<double, std::milli>(end - start).count();
                            }
                        } else if (mode == "metal" && batch == 1) {
                            id<MTLCommandQueue> queue = [device newCommandQueue];
                            NSError* err = nil;
                            id<MTLLibrary> lib = [device newDefaultLibrary];
                            if (!lib) continue;
                            id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
                            if (!func) continue;
                            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                            if (!pipeline) continue;
                            id<MTLBuffer> a_buf = [device newBufferWithBytes:a[0].data() length:sizeof(float)*m*k options:MTLResourceStorageModeShared];
                            id<MTLBuffer> b_buf = [device newBufferWithBytes:b[0].data() length:sizeof(float)*k*n options:MTLResourceStorageModeShared];
                            id<MTLBuffer> c_buf = [device newBufferWithLength:sizeof(float)*m*n options:MTLResourceStorageModeShared];
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
                                total_ms += std::chrono::duration<double, std::milli>(end - start).count();
                            }
                        } else if (mode == "metal_batched" && batch > 1) {
                            id<MTLCommandQueue> queue = [device newCommandQueue];
                            NSError* err = nil;
                            id<MTLLibrary> lib = [device newDefaultLibrary];
                            if (!lib) continue;
                            id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
                            if (!func) continue;
                            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                            if (!pipeline) continue;
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
                                if (!lib) { skipped = true; } else {
                                    id<MTLFunction> func = [lib newFunctionWithName:@"gemm_fp16"];
                                    if (!func) { skipped = true; } else {
                                        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                                        if (!pipeline) { skipped = true; } else {
                                            id<MTLBuffer> a_buf = [device newBufferWithBytes:a16.data() length:sizeof(half)*m*k options:MTLResourceStorageModePrivate];
                                            id<MTLBuffer> b_buf = [device newBufferWithBytes:b16.data() length:sizeof(half)*k*n options:MTLResourceStorageModePrivate];
                                            id<MTLBuffer> c_buf = [device newBufferWithLength:sizeof(half)*m*n options:MTLResourceStorageModePrivate];
                                            for (int r = 0; r < repeats; ++r) {
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
                                skipped = true;
                            }
                        
                            if (batch != 1) { skipped = true; } else {
                                id<MTLCommandQueue> queue = [device newCommandQueue];
                                NSError* err = nil;
                                id<MTLLibrary> lib = [device newDefaultLibrary];
                                if (!lib) { skipped = true; } else {
                                    id<MTLFunction> func = [lib newFunctionWithName:@"gemm_tiled"];
                                    if (!func) { skipped = true; } else {
                                        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                                        if (!pipeline) { skipped = true; } else {
                                            id<MTLBuffer> a_buf = [device newBufferWithBytes:a[0].data() length:sizeof(float)*m*k options:MTLResourceStorageModePrivate];
                                            id<MTLBuffer> b_buf = [device newBufferWithBytes:b[0].data() length:sizeof(float)*k*n options:MTLResourceStorageModePrivate];
                                            id<MTLBuffer> c_buf = [device newBufferWithLength:sizeof(float)*m*n options:MTLResourceStorageModePrivate];
                                            for (int r = 0; r < repeats; ++r) {
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
                            if (!lib) continue;
                            id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
                            if (!func) continue;
                            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                            if (!pipeline) continue;
                            for (int r = 0; r < repeats; ++r) {
                                std::vector<id<MTLBuffer>> a_bufs(batch);
                                std::vector<id<MTLBuffer>> b_bufs(batch);
                                std::vector<id<MTLBuffer>> c_bufs(batch);
                                for (int bidx = 0; bidx < batch; ++bidx) {
                                    a_bufs[bidx] = [device newBufferWithBytes:a[bidx].data() length:sizeof(float)*m*k options:MTLResourceStorageModeShared];
                                    b_bufs[bidx] = [device newBufferWithBytes:b[bidx].data() length:sizeof(float)*k*n options:MTLResourceStorageModeShared];
                                    c_bufs[bidx] = [device newBufferWithLength:sizeof(float)*m*n options:MTLResourceStorageModeShared];
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
                        } else {
                            skipped = true;
                        }
                        if (total_ms > 0.0) {
                            avg_ms = total_ms / repeats;
                        }
                        char size_str[32];
                        snprintf(size_str, sizeof(size_str), "%dx%dx%d", m, n, k);
                        std::time_t t = std::time(nullptr);
                        char timebuf[32];
                        std::strftime(timebuf, sizeof(timebuf), "%FT%T%z", std::localtime(&t));
                        csv << timebuf << "," << commit << ",GEMM," << mode << "," << device_csv << "," << size_str << "," << batch << "," << avg_ms << "," << (skipped ? "skipped" : "ok") << "\n";
                        std::cout << "[BENCHMARK][" << device_csv << "] " << mode << " size=" << size_str << " batch=" << batch << ": " << (skipped ? "skipped" : std::to_string(avg_ms) + " ms") << std::endl;
                    }
                }
            }
        }
            NSString* device_name = [device name];
            bool isNeuralEngine = false;
            if ([device_name containsString:@"ANE"] || [device_name containsString:@"Neural"]) {
                isNeuralEngine = true;
            }
            std::cout << "[GEMM-BENCH] Device: " << [device_name UTF8String] << std::endl;
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

                // --- Batched GEMM Benchmarks (Batch Size = 8) ---
                int batch = 8;
                std::vector<std::vector<float>> batch_a(batch, std::vector<float>(m * k));
                std::vector<std::vector<float>> batch_b(batch, std::vector<float>(k * n));
                std::vector<std::vector<float>> batch_c(batch, std::vector<float>(m * n));
                for (int b = 0; b < batch; ++b) {
                    for (auto& v : batch_a[b]) v = dist(gen);
                    for (auto& v : batch_b[b]) v = dist(gen);
                }
                // --- CPU (GCD Batched) ---
                double gcd_batch_total = 0.0;
                for (int r = 0; r < repeats; ++r) {
                    auto start = std::chrono::high_resolution_clock::now();
                    dispatch_apply(batch, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t b) {
                        gemm_gcd(batch_a[b].data(), batch_b[b].data(), batch_c[b].data(), m, n, k);
                    });
                    auto end = std::chrono::high_resolution_clock::now();
                    gcd_batch_total += std::chrono::duration<double, std::milli>(end - start).count();
                }
                double gcd_batch_ms = gcd_batch_total / repeats;
                std::cout << "[BENCHMARK] CPU (GCD Batched, batch=" << batch << "): " << gcd_batch_ms << " ms" << std::endl;
                // --- CPU (Accelerate+GCD Batched) ---
                double accgcd_batch_total = 0.0;
                for (int r = 0; r < repeats; ++r) {
                    auto start = std::chrono::high_resolution_clock::now();
                    dispatch_apply(batch, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t b) {
                        gemm_accelerate(batch_a[b].data(), batch_b[b].data(), batch_c[b].data(), m, n, k);
                    });
                    auto end = std::chrono::high_resolution_clock::now();
                    accgcd_batch_total += std::chrono::duration<double, std::milli>(end - start).count();
                }
                double accgcd_batch_ms = accgcd_batch_total / repeats;
                std::cout << "[BENCHMARK] CPU (Accelerate+GCD Batched, batch=" << batch << "): " << accgcd_batch_ms << " ms" << std::endl;

                // --- Metal ---
                // --- Metal ---
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
                    std::cerr << "[GEMM-BENCH] Pipeline-Fehler auf Device: " << [device_name UTF8String] << std::endl;
                    continue;
                }
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
                double max_diff = 0.0;
                for (size_t i = 0; i < c_metal.size(); ++i)
                    max_diff = std::max(max_diff, static_cast<double>(std::abs(c_metal[i] - c_ref[i])));
                char size_str[32];
                snprintf(size_str, sizeof(size_str), "%dx%dx%d", m, n, k);
                std::time_t t = std::time(nullptr);
                char timebuf[32];
                std::strftime(timebuf, sizeof(timebuf), "%FT%T%z", std::localtime(&t));
                if (max_diff < 1e-4) {
                    std::cout << "[BENCHMARK][" << [device_name UTF8String] << "] CPU: " << cpu_ms << " ms, Metal: " << metal_ms << " ms, Speedup: " << (cpu_ms / metal_ms) << std::endl;
                }

                // --- Metal Batched GEMM (Batch Size = 8, Multiple Command Buffers) ---
                double metal_batch_total = 0.0;
                for (int r = 0; r < repeats; ++r) {
                    std::vector<id<MTLBuffer>> a_bufs(batch);
                    std::vector<id<MTLBuffer>> b_bufs(batch);
                    std::vector<id<MTLBuffer>> c_bufs(batch);
                    std::vector<std::vector<float>> c_metal_batch(batch, std::vector<float>(m * n));
                    for (int b = 0; b < batch; ++b) {
                        a_bufs[b] = [device newBufferWithBytes:batch_a[b].data() length:sizeof(float)*m*k options:MTLResourceStorageModeShared];
                        b_bufs[b] = [device newBufferWithBytes:batch_b[b].data() length:sizeof(float)*k*n options:MTLResourceStorageModeShared];
                        c_bufs[b] = [device newBufferWithLength:sizeof(float)*m*n options:MTLResourceStorageModeShared];
                    }
                    std::vector<id<MTLCommandBuffer>> cmd_bufs(batch);
                    std::vector<id<MTLComputeCommandEncoder>> encoders(batch);
                    auto start = std::chrono::high_resolution_clock::now();
                    for (int b = 0; b < batch; ++b) {
                        cmd_bufs[b] = [queue commandBuffer];
                        encoders[b] = [cmd_bufs[b] computeCommandEncoder];
                        [encoders[b] setComputePipelineState:pipeline];
                        [encoders[b] setBuffer:a_bufs[b] offset:0 atIndex:0];
                        [encoders[b] setBuffer:b_bufs[b] offset:0 atIndex:1];
                        [encoders[b] setBuffer:c_bufs[b] offset:0 atIndex:2];
                        int dims[3] = {m, n, k};
                        [encoders[b] setBytes:&dims length:sizeof(dims) atIndex:3];
                        MTLSize grid = MTLSizeMake(m, n, 1);
                        MTLSize threadgroup = MTLSizeMake(8, 8, 1);
                        [encoders[b] dispatchThreads:grid threadsPerThreadgroup:threadgroup];
                        [encoders[b] endEncoding];
                        [cmd_bufs[b] commit];
                    }
                    for (int b = 0; b < batch; ++b) {
                        [cmd_bufs[b] waitUntilCompleted];
                        memcpy(c_metal_batch[b].data(), [c_bufs[b] contents], sizeof(float)*m*n);
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    metal_batch_total += std::chrono::duration<double, std::milli>(end - start).count();
                }
                double metal_batch_ms = metal_batch_total / repeats;
                std::cout << "[BENCHMARK][" << [device_name UTF8String] << "] Metal Batched (batch=" << batch << "): " << metal_batch_ms << " ms" << std::endl;
                    csv << timebuf << "," << commit << "," << operator_name << "," << (isNeuralEngine ? "NeuralEngine" : "MetalGPU") << "," << size_str << "," << cpu_ms << "," << metal_ms << "," << (cpu_ms / metal_ms) << "\n";
                } else {
                    std::cerr << "[GEMM-BENCH] Fehler! MaxDiff: " << max_diff << std::endl;
                }
            }
        }
        return 0;
    }
}
