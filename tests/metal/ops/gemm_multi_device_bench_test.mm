#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
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
extern "C" void gemm_cpu(const float* a, const float* b, float* c, int m, int n, int k) {
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
extern "C" void gemm_accelerate(const float* a, const float* b, float* c, int m, int n, int k) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
}

// GCD parallel GEMM (naive, each row)
extern "C" void gemm_gcd(const float* a, const float* b, float* c, int m, int n, int k) {
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

bool minimal_gemm_test() {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = @[MTLCreateSystemDefaultDevice()];
        if (!devices || [devices count] == 0) {
            std::cerr << "[TEST] No Metal devices found!" << std::endl;
            std::cerr << std::flush;
            return false;
        }
        int m = 2, n = 2, k = 2;
        float a[4] = {1, 2, 3, 4};
        float b[4] = {5, 6, 7, 8};
        float c[4] = {0, 0, 0, 0};
        gemm_cpu(a, b, c, m, n, k);
        float expected[4] = {19, 22, 43, 50};
        for (int i = 0; i < 4; ++i) {
            if (std::abs(c[i] - expected[i]) > 1e-3) {
                std::cerr << "[TEST] gemm_cpu failed: c[" << i << "]=" << c[i] << ", expected=" << expected[i] << std::endl;
                std::cerr << std::flush;
                return false;
            }
        }
        std::cout << "[TEST] gemm_cpu minimal test passed." << std::endl;
        std::cout << std::flush;
        return true;
    }
}

int main() {
    std::cout << "[DEBUG] Entering main()" << std::endl;
    std::cout << std::flush;
    if (!minimal_gemm_test()) {
        std::cerr << "[ERROR] Minimal GEMM test failed. Aborting benchmark." << std::endl;
        std::cerr << std::flush;
        return 1;
    }
    std::cout << "[DEBUG] Minimal test completed. Starting full benchmark..." << std::endl;
    std::cout << std::flush;
    @autoreleasepool {
        auto bench_start = std::chrono::steady_clock::now();
        NSArray<id<MTLDevice>>* devices = @[MTLCreateSystemDefaultDevice()];
if (!devices || [devices count] == 0) {
    std::cerr << "[ERROR] No Metal devices found or device enumeration failed!" << std::endl;
    return 1;
}
std::cout << "[DEBUG] devices count = " << [devices count] << std::endl;
        for (NSUInteger i = 0; i < [devices count]; ++i) {
            std::cout << "[DEBUG] about to access device " << i << std::endl;
            id<MTLDevice> dev = [devices objectAtIndex:i];
            std::cout << "[DEBUG] got device " << i << std::endl;
            std::cout << "[DEBUG] about to access device name for device " << i << std::endl;
            std::cout << "[DEBUG] device[" << i << "] name: " << [[dev name] UTF8String] << std::endl;
            std::cout << "[DEBUG] printed device name for device " << i << std::endl;
        }
        const char* commit = "f35fc96e";
        const char* operator_name = "GEMM";
        const char* csv_file = "benchmarks_ops.csv";
        std::vector<std::tuple<int,int,int>> sizes = { {32,32,32} };
        std::vector<int> batch_sizes = {1};
        std::vector<std::string> modes = {"cpu_naive", "cpu_accelerate"};
        std::ifstream check(csv_file);
        bool file_exists = check.good();
        check.close();
        std::ofstream csv(csv_file, std::ios::app);
        if (!file_exists) {
            csv << "timestamp,commit,operator,mode,device,size,batch,avg_ms,min_ms,max_ms,stddev_ms,run_type,repetition,status\n";
        }
        bool timed_out = false;
        std::cout << "[DEBUG] sizes.size() = " << sizes.size() << std::endl;
        for (size_t i = 0; i < sizes.size(); ++i) {
            std::cout << "[DEBUG] sizes[" << i << "] = (" << std::get<0>(sizes[i]) << "," << std::get<1>(sizes[i]) << "," << std::get<2>(sizes[i]) << ")" << std::endl;
        }
        std::cout << "[DEBUG] batch_sizes.size() = " << batch_sizes.size() << std::endl;
        for (size_t i = 0; i < batch_sizes.size(); ++i) {
            std::cout << "[DEBUG] batch_sizes[" << i << "] = " << batch_sizes[i] << std::endl;
        }
        for (id<MTLDevice> device in devices) {
            NSString* device_name = [device name];
            std::cout << "[GEMM-BENCH] Device: " << [device_name UTF8String] << std::endl;
            std::cout << "[DEBUG] After device name print" << std::endl << std::flush;
            break; // Exit after first device name print for minimal test
            // for (const auto& sz : sizes) {
            // ... rest of benchmark logic commented out ...
        
                now = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - bench_start).count();
                if (elapsed > 600) { std::cout << "[TIMEOUT] 10 minute limit reached. Aborting benchmark." << std::endl; std::cout << std::flush; timed_out = true; break; }
                int m = std::get<0>(sz), n = std::get<1>(sz), k = std::get<2>(sz);
                std::cout << "[DEBUG] m=" << m << " n=" << n << " k=" << k << std::endl << std::flush;
                std::cout << "  Size: " << m << "x" << n << "x" << k << std::endl; std::cout << std::flush;
                for (int batch : batch_sizes) {
                    std::cout << "[DEBUG] batch=" << batch << std::endl << std::flush;
                    now = std::chrono::steady_clock::now();
                    elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - bench_start).count();
                    if (elapsed > 600) { std::cout << "[TIMEOUT] 10 minute limit reached. Aborting benchmark." << std::endl; std::cout << std::flush; timed_out = true; break; }
                    std::cout << "    Batch: " << batch << std::endl; std::cout << std::flush;
                    std::cout << "[DEBUG] Allocating matrices" << std::endl << std::flush;
                    std::vector<std::vector<float>> a(batch, std::vector<float>(m * k));
                    std::vector<std::vector<float>> b(batch, std::vector<float>(k * n));
                    std::vector<std::vector<float>> c(batch, std::vector<float>(m * n));
                    std::cout << "[DEBUG] Matrices allocated" << std::endl << std::flush;
                    std::mt19937 gen(42);
                    std::uniform_real_distribution<float> dist(-1, 1);
                    std::cout << "[DEBUG] Initializing matrices" << std::endl << std::flush;
                    for (int bidx = 0; bidx < batch; ++bidx) {
                        for (auto& v : a[bidx]) v = dist(gen);
                        for (auto& v : b[bidx]) v = dist(gen);
                        std::fill(c[bidx].begin(), c[bidx].end(), 0.0f);
                    }
                    std::cout << "[DEBUG] Matrices initialized" << std::endl << std::flush;
                    for (const auto& mode : modes) {
                        now = std::chrono::steady_clock::now();
                        elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - bench_start).count();
                        if (elapsed > 600) { std::cout << "[TIMEOUT] 10 minute limit reached. Aborting benchmark." << std::endl; std::cout << std::flush; timed_out = true; break; }
                        std::cout << "      Mode: " << mode << std::endl; std::cout << std::flush;
                        for (const std::string& run_type : {std::string("short"), std::string("longterm")}) {
                            int repetitions = (run_type == "short") ? 10 : 100;
                            std::vector<double> timings;
                            std::string status = "ok";
                            try {
                                bool supported = true;
                                double total_ms = 0.0;
                                timings.clear();
                                // --- Mode selection logic ---
                                if ((mode == "cpu_naive" || mode == "cpu_accelerate") && batch != 1) supported = false;
                                if ((mode == "metal" || mode == "metal_fp16" || mode == "mps" || mode == "mps_fp16") && batch != 1) supported = false;
                                if ((mode == "metal_batched" || mode == "metal_tiled") && batch == 1) supported = false;
                                if ((mode == "ane" || mode == "hybrid") && !isNeuralEngine) supported = false;
                                // (Add more device/mode checks as needed)
                                if (!supported) status = "skipped";
                                if (status == "ok") {
                                    for (int r = 0; r < repetitions; ++r) {
                                        auto start = std::chrono::high_resolution_clock::now();
                                        // --- Insert actual GEMM dispatch for mode here (as in previous logic) ---
                                        // For demonstration, call gemm_cpu for cpu_naive, etc.
                                        if (mode == "cpu_naive") {
                                            std::cout << "[DEBUG] Calling gemm_cpu: a=" << static_cast<const void*>(a[0].data()) << ", b=" << static_cast<const void*>(b[0].data()) << ", c=" << static_cast<void*>(c[0].data()) << std::endl << std::flush;
                                            gemm_cpu(a[0].data(), b[0].data(), const_cast<float*>(c[0].data()), m, n, k);
                                            std::cout << "[DEBUG] gemm_cpu done" << std::endl << std::flush;
                                        } else if (mode == "cpu_accelerate") {
                                            std::cout << "[DEBUG] Calling gemm_accelerate: a=" << static_cast<const void*>(a[0].data()) << ", b=" << static_cast<const void*>(b[0].data()) << ", c=" << static_cast<void*>(c[0].data()) << std::endl << std::flush;
                                            gemm_accelerate(a[0].data(), b[0].data(), c[0].data(), m, n, k);
                                            std::cout << "[DEBUG] gemm_accelerate done" << std::endl << std::flush;
                                        }
                                        // ... (add similar debug prints for other modes as needed)
                                    }
                                }

            for (const auto& sz : sizes) {
                now = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - bench_start).count();
                if (elapsed > 600) { std::cout << "[TIMEOUT] 10 minute limit reached. Aborting benchmark." << std::endl; std::cout << std::flush; timed_out = true; break; }
                int m = std::get<0>(sz), n = std::get<1>(sz), k = std::get<2>(sz);
std::cout << "[DEBUG] m=" << m << " n=" << n << " k=" << k << std::endl << std::flush;
std::cout << "  Size: " << m << "x" << n << "x" << k << std::endl; std::cout << std::flush;
for (int batch : batch_sizes) {
    std::cout << "[DEBUG] batch=" << batch << std::endl << std::flush;
                    now = std::chrono::steady_clock::now();
                    elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - bench_start).count();
                    if (elapsed > 600) { std::cout << "[TIMEOUT] 10 minute limit reached. Aborting benchmark." << std::endl; std::cout << std::flush; timed_out = true; break; }
                    std::cout << "    Batch: " << batch << std::endl; std::cout << std::flush;
                    // --- Matrix initialization ---
                    std::vector<std::vector<float>> a(batch, std::vector<float>(m * k));
                    std::vector<std::vector<float>> b(batch, std::vector<float>(k * n));
                    std::vector<std::vector<float>> c(batch, std::vector<float>(m * n));
                    std::mt19937 gen(42);
                    std::uniform_real_distribution<float> dist(-1, 1);
                    for (int bidx = 0; bidx < batch; ++bidx) {
                        for (auto& v : a[bidx]) v = dist(gen);
                        for (auto& v : b[bidx]) v = dist(gen);
                        std::fill(c[bidx].begin(), c[bidx].end(), 0.0f);
                    }
                    for (const auto& mode : modes) {
                        now = std::chrono::steady_clock::now();
                        elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - bench_start).count();
                        if (elapsed > 600) { std::cout << "[TIMEOUT] 10 minute limit reached. Aborting benchmark." << std::endl; std::cout << std::flush; timed_out = true; break; }
                        std::cout << "      Mode: " << mode << std::endl; std::cout << std::flush;
                        for (const std::string& run_type : {std::string("short"), std::string("longterm")}) {
                            int repetitions = (run_type == "short") ? 10 : 100;
                            std::vector<double> timings;
                            std::string status = "ok";
                            try {
                                bool supported = true;
                                double total_ms = 0.0;
                                timings.clear();
                                // --- Mode selection logic ---
                                if ((mode == "cpu_naive" || mode == "cpu_accelerate") && batch != 1) supported = false;
                                if ((mode == "metal" || mode == "metal_fp16" || mode == "mps" || mode == "mps_fp16") && batch != 1) supported = false;
                                if ((mode == "metal_batched" || mode == "metal_tiled") && batch == 1) supported = false;
                                if ((mode == "ane" || mode == "hybrid") && !isNeuralEngine) supported = false;
                                // (Add more device/mode checks as needed)
                                if (!supported) status = "skipped";
                                if (status == "ok") {
                                    for (int r = 0; r < repetitions; ++r) {
                                        auto start = std::chrono::high_resolution_clock::now();
                                        // --- Insert actual GEMM dispatch for mode here (as in previous logic) ---
                                        // For demonstration, call gemm_cpu for cpu_naive, etc.
                                        if (mode == "cpu_naive") {
    std::cout << "[DEBUG] Calling gemm_cpu: a=" << static_cast<const void*>(a[0].data()) << ", b=" << static_cast<const void*>(b[0].data()) << ", c=" << static_cast<void*>(c[0].data()) << std::endl << std::flush;
    gemm_cpu(a[0].data(), b[0].data(), const_cast<float*>(c[0].data()), m, n, k);
    std::cout << "[DEBUG] gemm_cpu done" << std::endl << std::flush;
} else if (mode == "cpu_accelerate") {
    std::cout << "[DEBUG] Calling gemm_accelerate: a=" << static_cast<const void*>(a[0].data()) << ", b=" << static_cast<const void*>(b[0].data()) << ", c=" << static_cast<void*>(c[0].data()) << std::endl << std::flush;
    gemm_accelerate(a[0].data(), b[0].data(), c[0].data(), m, n, k);
    std::cout << "[DEBUG] gemm_accelerate done" << std::endl << std::flush;
}
                                        } else if (mode == "cpu_gcd") {
                                            dispatch_apply(batch, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t bidx) {
                                                gemm_gcd(a[bidx].data(), b[bidx].data(), const_cast<float*>(c[bidx].data()), m, n, k);
                                            });
                                        } else if (mode == "cpu_accgcd") {
                                            dispatch_apply(batch, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t bidx) {
                                                gemm_accelerate(a[bidx].data(), b[bidx].data(), const_cast<float*>(c[bidx].data()), m, n, k);
                                            });
                                        } else if (mode == "metal" && batch == 1) {
                                            id<MTLCommandQueue> queue = [device newCommandQueue];
                                            NSError* err = nil;
                                            id<MTLLibrary> lib = [device newDefaultLibrary];
                                            if (!lib) { status = "skipped"; break; }
                                            id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
                                            if (!func) { status = "skipped"; break; }
                                            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                                            if (!pipeline) { status = "skipped"; break; }
                                            id<MTLBuffer> a_buf = [device newBufferWithBytes:a[0].data() length:sizeof(float)*m*k options:MTLResourceStorageModeShared];
                                            id<MTLBuffer> b_buf = [device newBufferWithBytes:b[0].data() length:sizeof(float)*k*n options:MTLResourceStorageModeShared];
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
                                        }
                                        // ... (rest of GEMM mode dispatches)
                                    }
                                }
                                // ... (timing, CSV logging, etc.)
                            } catch (...) {
                                status = "error";
                            }
                        }
                        if (timed_out) break;
                    }
                    if (timed_out) break;
                }
                if (timed_out) break;
            }
            if (timed_out) break;
        }
        if (timed_out) {
            std::cout << "[BENCHMARK] Aborted due to timeout." << std::endl;
            std::cout << std::flush;
        } else {
            std::cout << "[BENCHMARK] Completed all tests." << std::endl;
            std::cout << std::flush;
        }
    }
    return 0;

        const char* operator_name = "GEMM";
        const char* csv_file = "benchmarks_ops.csv";
        std::vector<std::tuple<int,int,int>> sizes = { {32,32,32} };
        std::vector<int> batch_sizes = {1};
        std::vector<std::string> modes = {"cpu_naive", "cpu_accelerate"};
        std::ifstream check(csv_file);
        bool file_exists = check.good();
        check.close();
        std::ofstream csv(csv_file, std::ios::app);
        if (!file_exists) {
            csv << "timestamp,commit,operator,mode,device,size,batch,avg_ms,min_ms,max_ms,stddev_ms,run_type,repetition,status\n";
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
                    // --- Matrix initialization ---
                    std::vector<std::vector<float>> a(batch, std::vector<float>(m * k));
                    std::vector<std::vector<float>> b(batch, std::vector<float>(k * n));
                    std::vector<std::vector<float>> c(batch, std::vector<float>(m * n));
                    std::mt19937 gen(42);
                    std::uniform_real_distribution<float> dist(-1, 1);
                    for (int bidx = 0; bidx < batch; ++bidx) {
                        for (auto& v : a[bidx]) v = dist(gen);
                        for (auto& v : b[bidx]) v = dist(gen);
                        std::fill(c[bidx].begin(), c[bidx].end(), 0.0f);
                    }

                    for (const auto& mode : modes) {
                        for (const std::string& run_type : {std::string("short"), std::string("longterm")}) {
                            int repetitions = (run_type == "short") ? 10 : 100;
                            std::vector<double> timings;
                            std::string status = "ok";
                            try {
                                bool supported = true;
                                double total_ms = 0.0;
                                timings.clear();
                                // --- Mode selection logic ---
                                if ((mode == "cpu_naive" || mode == "cpu_accelerate") && batch != 1) supported = false;
                                if ((mode == "metal" || mode == "metal_fp16" || mode == "mps" || mode == "mps_fp16") && batch != 1) supported = false;
                                if ((mode == "metal_batched" || mode == "metal_tiled") && batch == 1) supported = false;
                                if ((mode == "ane" || mode == "hybrid") && !isNeuralEngine) supported = false;
                                // (Add more device/mode checks as needed)
                                if (!supported) status = "skipped";
                                if (status == "ok") {
                                    for (int r = 0; r < repetitions; ++r) {
                                        auto start = std::chrono::high_resolution_clock::now();
                                        // --- Insert actual GEMM dispatch for mode here (as in previous logic) ---
                                        // For demonstration, call gemm_cpu for cpu_naive, etc.
                                        if (mode == "cpu_naive") {
    std::cout << "[DEBUG] Calling gemm_cpu: a=" << static_cast<const void*>(a[0].data()) << ", b=" << static_cast<const void*>(b[0].data()) << ", c=" << static_cast<void*>(c[0].data()) << std::endl << std::flush;
    gemm_cpu(a[0].data(), b[0].data(), const_cast<float*>(c[0].data()), m, n, k);
    std::cout << "[DEBUG] gemm_cpu done" << std::endl << std::flush;
} else if (mode == "cpu_accelerate") {
    std::cout << "[DEBUG] Calling gemm_accelerate: a=" << static_cast<const void*>(a[0].data()) << ", b=" << static_cast<const void*>(b[0].data()) << ", c=" << static_cast<void*>(c[0].data()) << std::endl << std::flush;
    gemm_accelerate(a[0].data(), b[0].data(), c[0].data(), m, n, k);
    std::cout << "[DEBUG] gemm_accelerate done" << std::endl << std::flush;
}
                                        } else if (mode == "cpu_gcd") {
                                            dispatch_apply(batch, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t bidx) {
                                                gemm_gcd(a[bidx].data(), b[bidx].data(), const_cast<float*>(c[bidx].data()), m, n, k);
                                            });
                                        } else if (mode == "cpu_accgcd") {
                                            dispatch_apply(batch, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t bidx) {
                                                gemm_accelerate(a[bidx].data(), b[bidx].data(), const_cast<float*>(c[bidx].data()), m, n, k);
                                            });
                                        } else if (mode == "metal" && batch == 1) {
                                            id<MTLCommandQueue> queue = [device newCommandQueue];
                                            NSError* err = nil;
                                            id<MTLLibrary> lib = [device newDefaultLibrary];
                                            if (!lib) { status = "skipped"; break; }
                                            id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
                                            if (!func) { status = "skipped"; break; }
                                            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                                            if (!pipeline) { status = "skipped"; break; }
                                            id<MTLBuffer> a_buf = [device newBufferWithBytes:a[0].data() length:sizeof(float)*m*k options:MTLResourceStorageModeShared];
                                            id<MTLBuffer> b_buf = [device newBufferWithBytes:b[0].data() length:sizeof(float)*k*n options:MTLResourceStorageModeShared];
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
                                        } else if (mode == "metal_batched" && batch > 1) {
                                            id<MTLCommandQueue> queue = [device newCommandQueue];
                                            NSError* err = nil;
                                            id<MTLLibrary> lib = [device newDefaultLibrary];
                                            if (!lib) { status = "skipped"; break; }
                                            id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
                                            if (!func) { status = "skipped"; break; }
                                            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                                            if (!pipeline) { status = "skipped"; break; }
                                            std::vector<id<MTLBuffer>> a_bufs(batch);
                                            std::vector<id<MTLBuffer>> b_bufs(batch);
                                            std::vector<id<MTLBuffer>> c_bufs(batch);
                                            std::vector<id<MTLCommandBuffer>> cmd_bufs(batch);
                                            std::vector<id<MTLComputeCommandEncoder>> encoders(batch);
                                            for (int bidx = 0; bidx < batch; ++bidx) {
                                                a_bufs[bidx] = [device newBufferWithBytes:a[bidx].data() length:sizeof(float)*m*k options:MTLResourceStorageModeShared];
                                                b_bufs[bidx] = [device newBufferWithBytes:b[bidx].data() length:sizeof(float)*k*n options:MTLResourceStorageModeShared];
                                                c_bufs[bidx] = [device newBufferWithLength:sizeof(float)*m*n options:MTLResourceStorageModeShared];
                                            }
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
                                        } else if (mode == "mps" && batch == 1) {
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
                                                id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
                                                [mpsGemm encodeToCommandBuffer:cmd_buf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
                                                [cmd_buf commit];
                                                [cmd_buf waitUntilCompleted];
                                            } @catch (NSException *exception) {
                                                status = "skipped";
                                                break;
                                            }
                                        } else if (mode == "metal_fp16" && batch == 1) {
                                            if (![device supportsFamily:MTLGPUFamilyMac2]) { status = "skipped"; break; }
                                            std::vector<__fp16> a16(m*k);
                                            std::vector<__fp16> b16(k*n);
                                            for (int i = 0; i < m*k; ++i) a16[i] = (__fp16)a[0][i];
                                            for (int i = 0; i < k*n; ++i) b16[i] = (__fp16)b[0][i];
                                            id<MTLCommandQueue> queue = [device newCommandQueue];
                                            NSError* err = nil;
                                            id<MTLLibrary> lib = [device newDefaultLibrary];
                                            if (!lib) { status = "skipped"; break; }
                                            id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel_fp16"];
                                            if (!func) { status = "skipped"; break; }
                                            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                                            if (!pipeline) { status = "skipped"; break; }
                                            id<MTLBuffer> a_buf = [device newBufferWithBytes:a16.data() length:sizeof(__fp16)*m*k options:MTLResourceStorageModePrivate];
                                            id<MTLBuffer> b_buf = [device newBufferWithBytes:b16.data() length:sizeof(__fp16)*k*n options:MTLResourceStorageModePrivate];
                                            id<MTLBuffer> c_buf = [device newBufferWithLength:sizeof(__fp16)*m*n options:MTLResourceStorageModePrivate];
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
                                        } else if (mode == "mps_fp16" && batch == 1) {
                                            if (![device supportsFamily:MTLGPUFamilyMac2]) { status = "skipped"; break; }
                                            std::vector<__fp16> a16(m*k);
                                            std::vector<__fp16> b16(k*n);
                                            for (int i = 0; i < m*k; ++i) a16[i] = (__fp16)a[0][i];
                                            for (int i = 0; i < k*n; ++i) b16[i] = (__fp16)b[0][i];
                                            id<MTLCommandQueue> queue = [device newCommandQueue];
                                            id<MTLBuffer> a_buf = [device newBufferWithBytes:a16.data() length:sizeof(__fp16)*m*k options:MTLResourceStorageModePrivate];
                                            id<MTLBuffer> b_buf = [device newBufferWithBytes:b16.data() length:sizeof(__fp16)*k*n options:MTLResourceStorageModePrivate];
                                            id<MTLBuffer> c_buf = [device newBufferWithLength:sizeof(__fp16)*m*n options:MTLResourceStorageModePrivate];
                                            MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k*sizeof(__fp16) dataType:MPSDataTypeFloat16];
                                            MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:n rowBytes:n*sizeof(__fp16) dataType:MPSDataTypeFloat16];
                                            MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n*sizeof(__fp16) dataType:MPSDataTypeFloat16];
                                            MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:a_buf descriptor:descA];
                                            MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:b_buf descriptor:descB];
                                            MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:c_buf descriptor:descC];
                                            MPSMatrixMultiplication *mpsGemm = [[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:NO resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];
                                            id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
                                            [mpsGemm encodeToCommandBuffer:cmd_buf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
                                            [cmd_buf commit];
                                            [cmd_buf waitUntilCompleted];
                                        } else if (mode == "ane" && isNeuralEngine) {
                                            // For demonstration, treat as Metal on ANE device (real ANE dispatch would be CoreML/ANE APIs)
                                            id<MTLCommandQueue> queue = [device newCommandQueue];
                                            NSError* err = nil;
                                            id<MTLLibrary> lib = [device newDefaultLibrary];
                                            if (!lib) { status = "skipped"; break; }
                                            id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
                                            if (!func) { status = "skipped"; break; }
                                            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                                            if (!pipeline) { status = "skipped"; break; }
                                            id<MTLBuffer> a_buf = [device newBufferWithBytes:a[0].data() length:sizeof(float)*m*k options:MTLResourceStorageModeShared];
                                            id<MTLBuffer> b_buf = [device newBufferWithBytes:b[0].data() length:sizeof(float)*k*n options:MTLResourceStorageModeShared];
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
                                        } else if (mode == "hybrid" && isNeuralEngine) {
                                            // For demonstration, split batch between CPU and Metal (real hybrid would be more complex)
                                            int split = batch / 2;
                                            // CPU half
                                            dispatch_apply(split, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t bidx) {
                                                gemm_cpu(a[bidx].data(), b[bidx].data(), const_cast<float*>(c[bidx].data()), m, n, k);
                                            });
                                            // Metal half
                                            id<MTLCommandQueue> queue = [device newCommandQueue];
                                            NSError* err = nil;
                                            id<MTLLibrary> lib = [device newDefaultLibrary];
                                            if (!lib) { status = "skipped"; break; }
                                            id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
                                            if (!func) { status = "skipped"; break; }
                                            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                                            if (!pipeline) { status = "skipped"; break; }
                                            for (int bidx = split; bidx < batch; ++bidx) {
                                                id<MTLBuffer> a_buf = [device newBufferWithBytes:a[bidx].data() length:sizeof(float)*m*k options:MTLResourceStorageModeShared];
                                                id<MTLBuffer> b_buf = [device newBufferWithBytes:b[bidx].data() length:sizeof(float)*k*n options:MTLResourceStorageModeShared];
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
                                            }
                                        } else {
                                            status = "skipped";
                                            break;
                                        }
                                        auto end = std::chrono::high_resolution_clock::now();
                                        double ms = std::chrono::duration<double, std::milli>(end - start).count();
                                        timings.push_back(ms);
                                    }
                                }
                            } catch (...) {
                                status = "error";
                            }
                            // --- Compute stats ---
                            double avg_ms = 0, min_ms = 0, max_ms = 0, stddev_ms = 0;
                            if (!timings.empty()) {
                                avg_ms = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
                                min_ms = *std::min_element(timings.begin(), timings.end());
                                max_ms = *std::max_element(timings.begin(), timings.end());
                                double sum_sq = 0.0;
                                for (double t : timings) sum_sq += (t - avg_ms) * (t - avg_ms);
                                stddev_ms = sqrt(sum_sq / timings.size());
                            }
                            // --- Log to CSV ---
                            std::time_t t = std::time(nullptr);
                            char timebuf[32];
                            std::strftime(timebuf, sizeof(timebuf), "%FT%T%z", std::localtime(&t));
                            csv << timebuf << "," << commit << "," << operator_name << "," << mode << "," << [device_name UTF8String] << "," << m << "x" << n << "x" << k << "," << batch << ","
                                << avg_ms << "," << min_ms << "," << max_ms << "," << stddev_ms << "," << run_type << "," << repetitions << "," << status << "\n";
                        }
                    }

                }
            }
        }
        return 0;
    }
}
