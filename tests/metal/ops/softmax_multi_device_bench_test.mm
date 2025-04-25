#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <random>
#include <fstream>
#include <iostream>
#include <thread>
extern "C" bool softmax_metal_batch(const std::vector<const float*>& ins, std::vector<float*>& outs, int size);
namespace softmax_dispatch {
void softmax_cpu(const float* in, float* out, int size) {
    float maxval = *std::max_element(in, in + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        out[i] = std::exp(in[i] - maxval);
        sum += out[i];
    }
    for (int i = 0; i < size; ++i)
        out[i] /= sum;
}
void softmax_cpu_mt(const float* in, float* out, int size, int num_threads = 0) {
    // Für Softmax ist Multithreading auf einzelne Softmax-Vektoren nicht effizient,
    // aber für einen Batch von Vektoren schon. Hier als Platzhalter Single-Thread.
    softmax_cpu(in, out, size);
}
}
void bench_softmax(const std::string& mode, int size, int batch, std::ofstream& csv) {
    std::vector<std::vector<float>> ins(batch, std::vector<float>(size));
    std::vector<std::vector<float>> outs(batch, std::vector<float>(size));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < size; ++i)
            ins[b][i] = dist(rng);
    std::vector<const float*> ptr_in(batch);
    std::vector<float*> ptr_out(batch);
    for (int b = 0; b < batch; ++b) {
        ptr_in[b] = ins[b].data();
        ptr_out[b] = outs[b].data();
    }
    int repeat = 10;
    double total_ms = 0.0;
    for (int r = 0; r < repeat; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        if (mode == "CPU_SINGLE") {
            for (int b = 0; b < batch; ++b)
                softmax_dispatch::softmax_cpu(ins[b].data(), outs[b].data(), size);
        } else if (mode == "CPU_MULTI") {
            for (int b = 0; b < batch; ++b)
                softmax_dispatch::softmax_cpu_mt(ins[b].data(), outs[b].data(), size);
        } else if (mode == "METAL_BATCH") {
            softmax_metal_batch(ptr_in, ptr_out, size);
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
    std::ofstream csv("benchmarks_softmax_dispatch.csv");
    csv << "mode,size,batch,avg_ms,latency,throughput" << std::endl;
    std::vector<int> sizes = {128, 1024, 8192, 32768, 131072, 524288};
    std::vector<int> batches = {1, 8, 32, 128};
    for (auto size : sizes) {
        for (auto batch : batches) {
            bench_softmax("CPU_SINGLE", size, batch, csv);
            bench_softmax("CPU_MULTI", size, batch, csv);
            bench_softmax("METAL_BATCH", size, batch, csv);
        }
    }
    return 0;
}
