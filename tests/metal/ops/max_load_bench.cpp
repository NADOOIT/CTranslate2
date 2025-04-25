#include <chrono>
#include <vector>
#include <random>
#include <thread>
#include <iostream>
#include <numeric>
#include <atomic>
#include <cstring>
#include <algorithm>
#include <fstream>
extern "C" bool add_metal_batch(const std::vector<const float*>& a, const std::vector<const float*>& b, std::vector<float*>& c, int size);
extern "C" bool multiply_metal_batch(const std::vector<const float*>& a, const std::vector<const float*>& b, std::vector<float*>& c, int size);

namespace maxload {
void add_cpu_mt(const float* a, const float* b, float* c, int size, int num_threads) {
    std::vector<std::thread> threads(num_threads);
    int chunk = (size + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk;
        int end = std::min(start + chunk, size);
        threads[t] = std::thread([=]() {
            for (int i = start; i < end; ++i)
                c[i] = a[i] + b[i];
        });
    }
    for (auto& th : threads) th.join();
}
void multiply_cpu_mt(const float* a, const float* b, float* c, int size, int num_threads) {
    std::vector<std::thread> threads(num_threads);
    int chunk = (size + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk;
        int end = std::min(start + chunk, size);
        threads[t] = std::thread([=]() {
            for (int i = start; i < end; ++i)
                c[i] = a[i] * b[i];
        });
    }
    for (auto& th : threads) th.join();
}
}

void bench_max_cpu(const std::string& op, int size, int duration_s, int threads) {
    std::vector<float> a(size, 1.1f), b(size, 2.2f), c(size, 0.0f);
    int64_t ops = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = t0;
    do {
        if (op == "add")
            maxload::add_cpu_mt(a.data(), b.data(), c.data(), size, threads);
        else if (op == "multiply")
            maxload::multiply_cpu_mt(a.data(), b.data(), c.data(), size, threads);
        ++ops;
        t1 = std::chrono::high_resolution_clock::now();
    } while (std::chrono::duration<double>(t1 - t0).count() < duration_s);
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double total_elems = double(ops) * size;
    double throughput = total_elems / elapsed;
    std::cout << "[MAXLOAD][CPU_MT] op=" << op << " size=" << size << " threads=" << threads << " time=" << elapsed << "s throughput=" << throughput << " elem/s" << std::endl;
}

void bench_max_metal(const std::string& op, int size, int batch, int duration_s) {
    std::vector<std::vector<float>> as(batch, std::vector<float>(size, 1.1f));
    std::vector<std::vector<float>> bs(batch, std::vector<float>(size, 2.2f));
    std::vector<std::vector<float>> cs(batch, std::vector<float>(size, 0.0f));
    std::vector<const float*> ptr_a(batch), ptr_b(batch);
    std::vector<float*> ptr_c(batch);
    for (int b = 0; b < batch; ++b) {
        ptr_a[b] = as[b].data();
        ptr_b[b] = bs[b].data();
        ptr_c[b] = cs[b].data();
    }
    int64_t ops = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = t0;
    do {
        if (op == "add")
            add_metal_batch(ptr_a, ptr_b, ptr_c, size);
        else if (op == "multiply")
            multiply_metal_batch(ptr_a, ptr_b, ptr_c, size);
        ++ops;
        t1 = std::chrono::high_resolution_clock::now();
    } while (std::chrono::duration<double>(t1 - t0).count() < duration_s);
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double total_elems = double(ops) * size * batch;
    double throughput = total_elems / elapsed;
    std::cout << "[MAXLOAD][METAL_BATCH] op=" << op << " size=" << size << " batch=" << batch << " time=" << elapsed << "s throughput=" << throughput << " elem/s" << std::endl;
}

int main() {
    int duration = 10; // Sekunden
    int size = 4 * 1024 * 1024; // 4 Mio Elemente pro Array
    int batch = 128; // Für Metal
    int threads = std::thread::hardware_concurrency();
    std::cout << "==== MAX LOAD BENCH ====" << std::endl;
    bench_max_cpu("add", size * batch, duration, threads);
    bench_max_cpu("multiply", size * batch, duration, threads);
    bench_max_metal("add", size, batch, duration);
    bench_max_metal("multiply", size, batch, duration);
    return 0;
}
