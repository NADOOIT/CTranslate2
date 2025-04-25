#include <vector>
#include <thread>
#include <future>
#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>

extern "C" bool relu_metal_batch(const std::vector<const float*>& inputs, std::vector<float*>& outputs, int size);

void relu_cpu(const float* in, float* out, int size) {
    for (int i = 0; i < size; ++i)
        out[i] = std::max(in[i], 0.0f);
}

void relu_cpu_mt(const float* in, float* out, int size, int num_threads = 0) {
    if (num_threads <= 0)
        num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk = (size + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk;
        int end = std::min(start + chunk, size);
        threads.emplace_back([=]() {
            for (int i = start; i < end; ++i)
                out[i] = std::max(in[i], 0.0f);
        });
    }
    for (auto& th : threads) th.join();
}

void bench_and_log(const char* mode, int size, int batch, std::ofstream& csv) {
    std::vector<std::vector<float>> ins(batch, std::vector<float>(size));
    std::vector<std::vector<float>> outs(batch, std::vector<float>(size));
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-3, 3);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < size; ++i)
            ins[b][i] = dist(gen);
    std::vector<const float*> ptr_in(batch);
    std::vector<float*> ptr_out(batch);
    for (int b = 0; b < batch; ++b) {
        ptr_in[b] = ins[b].data();
        ptr_out[b] = outs[b].data();
    }
    const int repeats = 5;
    double total_ms = 0.0;
    for (int r = 0; r < repeats; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        if (strcmp(mode, "CPU_SINGLE") == 0) {
            for (int b = 0; b < batch; ++b)
                relu_cpu(ptr_in[b], ptr_out[b], size);
        } else if (strcmp(mode, "CPU_MULTI") == 0) {
            for (int b = 0; b < batch; ++b)
                relu_cpu_mt(ptr_in[b], ptr_out[b], size);
        } else if (strcmp(mode, "METAL_BATCH") == 0) {
            relu_metal_batch(ptr_in, ptr_out, size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double avg_ms = total_ms / repeats;
    double throughput = (batch * size) / (avg_ms * 1e3); // Elemente pro Mikrosekunde
    double latency = avg_ms / batch; // ms pro Array
    std::cout << "[BENCH] " << mode << " size=" << size << " batch=" << batch << " avg_ms=" << avg_ms << " latency=" << latency << "ms throughput=" << throughput << " elem/us" << std::endl;
    csv << mode << "," << size << "," << batch << "," << avg_ms << "," << latency << "," << throughput << "\n";
}

int main() {
    std::ofstream csv("benchmarks_adaptive_dispatch.csv");
    csv << "mode,size,batch,avg_ms,latency_ms,throughput_elem_per_us\n";
    std::vector<int> sizes = {128, 1024, 8192, 32768, 131072, 524288};
    std::vector<int> batches = {1, 8, 32, 128};
    for (int size : sizes) {
        for (int batch : batches) {
            bench_and_log("CPU_SINGLE", size, batch, csv);
            bench_and_log("CPU_MULTI", size, batch, csv);
            bench_and_log("METAL_BATCH", size, batch, csv);
        }
    }
    return 0;
}
