#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include <ctime>
#include "gemm_utils.h"
#include "gemm_cpu_accel.h"

#include "gemm_metal_runner.h"

int main() {
    @autoreleasepool {
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
        const char* commit = "f35fc96e";
        const char* operator_name = "GEMM";
        const char* variant = "Metal";
        const char* csv_file = "benchmarks_ops.csv";
        std::vector<std::tuple<int,int,int>> sizes = { {32,32,32}, {128,128,128}, {512,512,512} };
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
            fill_random(a, 42);
            fill_random(b, 43);
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
            // --- Benchmark Metal ---
            MetalGEMMResult result;
            try {
                result = run_metal_gemm(device, queue, a.data(), b.data(), c_metal.data(), c_ref.data(), m, n, k, repeats);
            } catch (const std::exception& ex) {
                std::cerr << "[GEMMTest] Metal error: " << ex.what() << std::endl;
                continue;
            }
            char size_str[32];
            snprintf(size_str, sizeof(size_str), "%dx%dx%d", m, n, k);
            std::time_t t = std::time(nullptr);
            char timebuf[32];
            std::strftime(timebuf, sizeof(timebuf), "%FT%T%z", std::localtime(&t));
            if (result.max_diff < 1e-4) {
                std::cout << "[GEMMTest] Erfolg! MaxDiff: " << result.max_diff << std::endl;
                csv << timebuf << "," << commit << "," << operator_name << "," << variant << "," << size_str << "," << cpu_ms << "," << result.metal_ms << "," << (cpu_ms / result.metal_ms) << "\n";
            } else {
                std::cerr << "[GEMMTest] Fehler! MaxDiff: " << result.max_diff << std::endl;
            }
        }
    }
    return 0;
}
