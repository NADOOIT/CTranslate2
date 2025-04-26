#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

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

#include <fstream>
#include <ctime>

int main() {
    @autoreleasepool {
        // Device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "[GEMMTest] Kein Metal-Device gefunden!" << std::endl;
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            std::cerr << "[GEMMTest] Keine CommandQueue!" << std::endl;
            return 2;
        }

        // Benchmark-Konstanten
        const char* commit = "f35fc96e";
        const char* operator_name = "GEMM";
        const char* variant = "Metal";
        const char* csv_file = "benchmarks_ops.csv";
        std::vector<std::tuple<int,int,int>> sizes = { {32,32,32}, {128,128,128}, {512,512,512} };

        // Schreibe Header, falls Datei neu
        std::ifstream check(csv_file);
        bool file_exists = check.good();
        check.close();
        std::ofstream csv(csv_file, std::ios::app);
        if (!file_exists) {
            csv << "timestamp,commit,operator,variant,size,cpu_ms,metal_ms,speedup\n";
        }

        for (const auto& sz : sizes) {
            int m = std::get<0>(sz), n = std::get<1>(sz), k = std::get<2>(sz);
            std::vector<float> a(m * k), b(k * n), c_ref(m * n), c_metal(m * n);
            std::mt19937 gen(42);
            std::uniform_real_distribution<float> dist(-1, 1);
            for (auto& v : a) v = dist(gen);
            for (auto& v : b) v = dist(gen);

            // --- Benchmark CPU ---
            const int repeats = 10;
            double cpu_total = 0.0;
            for (int r = 0; r < repeats; ++r) {
                auto start = std::chrono::high_resolution_clock::now();
                gemm_cpu(a.data(), b.data(), c_ref.data(), m, n, k);
                auto end = std::chrono::high_resolution_clock::now();
                cpu_total += std::chrono::duration<double, std::milli>(end - start).count();
            }
            double cpu_ms = cpu_total / repeats;

            // Metal-Buffer
            id<MTLBuffer> a_buf = [device newBufferWithBytes:a.data() length:sizeof(float)*a.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_buf = [device newBufferWithBytes:b.data() length:sizeof(float)*b.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> c_buf = [device newBufferWithLength:sizeof(float)*c_metal.size() options:MTLResourceStorageModeShared];
            if (!a_buf || !b_buf || !c_buf) {
                std::cerr << "[GEMMTest] Buffer-Fehler!" << std::endl;
                return 3;
            }

        // Pipeline laden
        NSError* err = nil;
        id<MTLLibrary> lib = [device newDefaultLibrary];
        if (!lib) {
            // Fallback: Versuche, explizit die metallib zu laden
            NSString* path = @"gemm_kernel.metallib";
            NSFileManager* fm = [NSFileManager defaultManager];
            if (![fm fileExistsAtPath:path]) {
                // Suche im Executable-Verzeichnis
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
                return 4;
            } else {
                std::cout << "[GEMMTest] Metal-Library explizit geladen: " << [path UTF8String] << std::endl;
            }
        }
        id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
        if (!func) {
            std::cerr << "[GEMMTest] Kein gemm_kernel!" << std::endl;
            return 5;
        }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
        if (!pipeline) {
            std::cerr << "[GEMMTest] Pipeline-Fehler: " << [[err localizedDescription] UTF8String] << std::endl;
            return 6;
        }

        // --- Benchmark Metal ---
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

        // Kommando-Buffer
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

        memcpy(c_metal.data(), [c_buf contents], sizeof(float)*c_metal.size());

        // Vergleich
        double max_diff = 0.0;
        memcpy(c_metal.data(), [c_buf contents], sizeof(float)*c_metal.size());
        for (size_t i = 0; i < c_metal.size(); ++i)
            max_diff = std::max(max_diff, static_cast<double>(std::abs(c_metal[i] - c_ref[i])));

        std::cout << "[BENCHMARK] CPU: " << cpu_ms << " ms, Metal: " << metal_ms << " ms, Speedup: " << (cpu_ms / metal_ms) << std::endl;
        char size_str[32];
        snprintf(size_str, sizeof(size_str), "%dx%dx%d", m, n, k);
        // Zeitstempel generieren
        std::time_t t = std::time(nullptr);
        char timebuf[32];
        std::strftime(timebuf, sizeof(timebuf), "%FT%T%z", std::localtime(&t));
        if (max_diff < 1e-4) {
            std::cout << "[GEMMTest] Erfolg! MaxDiff: " << max_diff << std::endl;
            csv << timebuf << "," << commit << "," << operator_name << "," << variant << "," << size_str << "," << cpu_ms << "," << metal_ms << "," << (cpu_ms / metal_ms) << "\n";
        } else {
            std::cerr << "[GEMMTest] Fehler! MaxDiff: " << max_diff << std::endl;
        }
