#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <random>
#include <fstream>
#include <iostream>
#include <thread>
extern "C" bool add_metal_batch(const std::vector<const float*>& a, const std::vector<const float*>& b, std::vector<float*>& c, int size);
namespace add_dispatch {
void add_cpu(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; ++i)
        c[i] = a[i] + b[i];
}
void add_cpu_mt(const float* a, const float* b, float* c, int size, int num_threads = 0) {
    if (num_threads <= 0)
        num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk = (size + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk;
        int end = std::min(start + chunk, size);
        threads.emplace_back([=]() {
            for (int i = start; i < end; ++i)
                c[i] = a[i] + b[i];
        });
    }
    for (auto& th : threads) th.join();
}
}
void bench_add(const std::string& mode, int size, int batch, std::ofstream& csv) {
    std::vector<std::vector<float>> as(batch, std::vector<float>(size));
    std::vector<std::vector<float>> bs(batch, std::vector<float>(size));
    std::vector<std::vector<float>> cs(batch, std::vector<float>(size));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < size; ++i) {
            as[b][i] = dist(rng);
            bs[b][i] = dist(rng);
        }
    std::vector<const float*> ptr_a(batch), ptr_b(batch);
    std::vector<float*> ptr_c(batch);
    for (int b = 0; b < batch; ++b) {
        ptr_a[b] = as[b].data();
        ptr_b[b] = bs[b].data();
        ptr_c[b] = cs[b].data();
    }
    int repeat = 10;
    double total_ms = 0.0;
    for (int r = 0; r < repeat; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        if (mode == "CPU_SINGLE") {
            for (int b = 0; b < batch; ++b)
                add_dispatch::add_cpu(as[b].data(), bs[b].data(), cs[b].data(), size);
        } else if (mode == "CPU_MULTI") {
            for (int b = 0; b < batch; ++b)
                add_dispatch::add_cpu_mt(as[b].data(), bs[b].data(), cs[b].data(), size);
        } else if (mode == "METAL_BATCH") {
            add_metal_batch(ptr_a, ptr_b, ptr_c, size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double avg_ms = total_ms / repeat;
    double latency = avg_ms / batch;
    double throughput = double(size * batch) / (avg_ms * 1000.0);
    csv << mode << "," << size << "," << batch << "," << avg_ms << "," << latency << "," << throughput << std::endl;
    std::cout << "[BENCH] " << mode << " size=" << size << " batch=" << batch << " avg_ms=" << avg_ms << " latency=" << latency << "ms throughput=" << throughput << " elem/us" << std::endl;
}
int main() {
    std::ofstream csv("benchmarks_add_dispatch.csv");
    csv << "mode,size,batch,avg_ms,latency,throughput" << std::endl;
    std::vector<int> sizes = {128, 1024, 8192, 32768, 131072, 524288};
    std::vector<int> batches = {1, 8, 32, 128};
    for (auto size : sizes) {
        for (auto batch : batches) {
            bench_add("CPU_SINGLE", size, batch, csv);
            bench_add("CPU_MULTI", size, batch, csv);
            bench_add("METAL_BATCH", size, batch, csv);
        }
    }
    return 0;
}
