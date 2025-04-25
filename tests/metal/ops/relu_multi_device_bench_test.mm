#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <vector>
#include <random>
#include <tuple>
#include <chrono>

// Referenz-ReLU (CPU)
void relu_cpu(const float* a, float* out, int size) {
    for (int i = 0; i < size; ++i) {
        out[i] = std::max(a[i], 0.0f);
    }
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
        const char* operator_name = "ReLU";
        const char* csv_file = "benchmarks_ops.csv";
        std::vector<int> sizes = { 1024, 16384, 65536, 262144 };
        std::ifstream check(csv_file);
        bool file_exists = check.good();
        check.close();
        std::ofstream csv(csv_file, std::ios::app);
        if (!file_exists) {
            csv << "timestamp,commit,operator,variant,size,cpu_ms,metal_ms,speedup\n";
        }
        for (id<MTLDevice> device in devices) {
            NSString* device_name = [device name];
            bool isNeuralEngine = false;
            if ([device_name containsString:@"ANE"] || [device_name containsString:@"Neural"]) {
                isNeuralEngine = true;
            }
            std::cout << "[ReLU-BENCH] Device: " << [device_name UTF8String] << std::endl;
            for (const auto& size : sizes) {
                std::vector<float> a(size), out_ref(size), out_metal(size);
                std::mt19937 gen(43);
                std::uniform_real_distribution<float> dist(-3, 3);
                for (auto& v : a) v = dist(gen);
                // --- CPU ---
                const int repeats = 10;
                double cpu_total = 0.0;
                for (int r = 0; r < repeats; ++r) {
                    auto start = std::chrono::high_resolution_clock::now();
                    relu_cpu(a.data(), out_ref.data(), size);
                    auto end = std::chrono::high_resolution_clock::now();
                    cpu_total += std::chrono::duration<double, std::milli>(end - start).count();
                }
                double cpu_ms = cpu_total / repeats;
                // --- Metal ---
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
                        std::cerr << "[ReLU-BENCH] Kein Metal-Library auf Device: " << [device_name UTF8String] << std::endl;
                        continue;
                    }
                }
                id<MTLFunction> func = [lib newFunctionWithName:@"relu_kernel"];
                if (!func) {
                    std::cerr << "[ReLU-BENCH] Kein relu_kernel auf Device: " << [device_name UTF8String] << std::endl;
                    continue;
                }
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
                if (!pipeline) {
                    std::cerr << "[ReLU-BENCH] Pipeline-Fehler auf Device: " << [device_name UTF8String] << std::endl;
                    continue;
                }
                id<MTLBuffer> a_buf = [device newBufferWithBytes:a.data() length:sizeof(float)*a.size() options:MTLResourceStorageModeShared];
                id<MTLBuffer> out_buf = [device newBufferWithLength:sizeof(float)*out_metal.size() options:MTLResourceStorageModeShared];
                if (!a_buf || !out_buf) {
                    std::cerr << "[ReLU-BENCH] Buffer-Fehler auf Device: " << [device_name UTF8String] << std::endl;
                    continue;
                }
                double metal_total = 0.0;
                for (int r = 0; r < repeats; ++r) {
                    id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:a_buf offset:0 atIndex:0];
                    [encoder setBuffer:out_buf offset:0 atIndex:1];
                    int sz = size;
                    [encoder setBytes:&sz length:sizeof(int) atIndex:2];
                    MTLSize grid = MTLSizeMake(size, 1, 1);
                    MTLSize threadgroup = MTLSizeMake(256, 1, 1);
                    [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
                    [encoder endEncoding];
                    auto start = std::chrono::high_resolution_clock::now();
                    [cmd_buf commit];
                    [cmd_buf waitUntilCompleted];
                    auto end = std::chrono::high_resolution_clock::now();
                    metal_total += std::chrono::duration<double, std::milli>(end - start).count();
                }
                double metal_ms = metal_total / repeats;
                memcpy(out_metal.data(), [out_buf contents], sizeof(float)*out_metal.size());
                double max_diff = 0.0;
                for (size_t i = 0; i < out_metal.size(); ++i)
                    max_diff = std::max(max_diff, static_cast<double>(std::abs(out_metal[i] - out_ref[i])));
                char size_str[32];
                snprintf(size_str, sizeof(size_str), "%d", size);
                std::time_t t = std::time(nullptr);
                char timebuf[32];
                std::strftime(timebuf, sizeof(timebuf), "%FT%T%z", std::localtime(&t));
                if (max_diff < 1e-4) {
                    std::cout << "[BENCHMARK][" << [device_name UTF8String] << "] CPU: " << cpu_ms << " ms, Metal: " << metal_ms << " ms, Speedup: " << (cpu_ms / metal_ms) << std::endl;
                    csv << timebuf << "," << commit << "," << operator_name << "," << (isNeuralEngine ? "NeuralEngine" : "MetalGPU") << "," << size_str << "," << cpu_ms << "," << metal_ms << "," << (cpu_ms / metal_ms) << "\n";
                } else {
                    std::cerr << "[ReLU-BENCH] Fehler! MaxDiff: " << max_diff << std::endl;
                }
            }
        }
        return 0;
    }
}
