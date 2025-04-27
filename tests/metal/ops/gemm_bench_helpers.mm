#include "gemm_bench_helpers.h"
#include "gemm_utils.h"
#include "gemm_cpu_accel.h"
#include "gemm_metal.h"
#include <random>
#include <iostream>

void run_cpu_batched_benchmarks(const std::vector<std::tuple<int,int,int>>& sizes,
                                const std::vector<int>& batch_sizes,
                                int repeats,
                                std::ofstream& csv,
                                const std::string& commit,
                                const std::string& operator_name,
                                id<MTLDevice> device,
                                const std::string& device_name) {
    for (const auto& sz : sizes) {
        int m = std::get<0>(sz), n = std::get<1>(sz), k = std::get<2>(sz);
        for (int batch : batch_sizes) {
            std::vector<std::vector<float>> a(batch, std::vector<float>(m * k));
            std::vector<std::vector<float>> b(batch, std::vector<float>(k * n));
            std::vector<std::vector<float>> c(batch, std::vector<float>(m * n));
            for (int bidx = 0; bidx < batch; ++bidx) {
                fill_random(a[bidx], 42 + bidx);
                fill_random(b[bidx], 142 + bidx);
            }
            double total_ms = 0.0;
            // CPU naive batched
            for (int r = 0; r < repeats; ++r) {
                auto start = std::chrono::high_resolution_clock::now();
                for (int bidx = 0; bidx < batch; ++bidx)
                    gemm_cpu(a[bidx].data(), b[bidx].data(), c[bidx].data(), m, n, k);
                auto end = std::chrono::high_resolution_clock::now();
                total_ms += std::chrono::duration<double, std::milli>(end - start).count();
            }
            double avg_ms = total_ms / repeats;
            std::string timebuf = current_time_string();
            char size_str[32];
            snprintf(size_str, sizeof(size_str), "%dx%dx%d", m, n, k);
            csv << timebuf << "," << commit << "," << operator_name << ",cpu_naive_batched," << device_name << "," << size_str << "," << batch << "," << avg_ms << "\n";
        }
    }
}

void run_metal_batched_benchmarks(const std::vector<std::tuple<int,int,int>>& sizes,
                                  const std::vector<int>& batch_sizes,
                                  int repeats,
                                  std::ofstream& csv,
                                  const std::string& commit,
                                  const std::string& operator_name,
                                  id<MTLDevice> device,
                                  const std::string& device_name) {
    for (const auto& sz : sizes) {
        int m = std::get<0>(sz), n = std::get<1>(sz), k = std::get<2>(sz);
        for (int batch : batch_sizes) {
            std::vector<std::vector<float>> a(batch, std::vector<float>(m * k));
            std::vector<std::vector<float>> b(batch, std::vector<float>(k * n));
            std::vector<std::vector<float>> c(batch, std::vector<float>(m * n));
            for (int bidx = 0; bidx < batch; ++bidx) {
                fill_random(a[bidx], 42 + bidx);
                fill_random(b[bidx], 142 + bidx);
            }
            NSError* err = nil;
            id<MTLLibrary> lib = [device newDefaultLibrary];
            if (!lib) continue;
            id<MTLFunction> func = [lib newFunctionWithName:@"gemm_kernel"];
            if (!func) continue;
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];
            if (!pipeline) continue;
            id<MTLCommandQueue> queue = [device newCommandQueue];
            double total_ms = 0.0;
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
                    memcpy(c[bidx].data(), [c_bufs[bidx] contents], sizeof(float)*m*n);
                }
                auto end = std::chrono::high_resolution_clock::now();
                total_ms += std::chrono::duration<double, std::milli>(end - start).count();
            }
            double avg_ms = total_ms / repeats;
            std::string timebuf = current_time_string();
            char size_str[32];
            snprintf(size_str, sizeof(size_str), "%dx%dx%d", m, n, k);
            csv << timebuf << "," << commit << "," << operator_name << ",metal_batched," << device_name << "," << size_str << "," << batch << "," << avg_ms << "\n";
        }
    }
}

