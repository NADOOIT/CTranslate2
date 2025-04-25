#include <gtest/gtest.h>
#include <vector>
#include <thread>
#include <future>
#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>

// Platzhalter für ReLU-Implementierungen
namespace relu_dispatch {

// CPU single-threaded
void relu_cpu(const float* in, float* out, int size) {
    for (int i = 0; i < size; ++i)
        out[i] = std::max(in[i], 0.0f);
}

// CPU multithreaded
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

// Metal/Batch-Dispatch (echte Implementierung)
extern "C" bool relu_metal_batch(const std::vector<const float*>& inputs, std::vector<float*>& outputs, int size);

// Adaptive Dispatcher
void relu_adaptive(const float* in, float* out, int size) {
    if (size < 4096) {
        relu_cpu(in, out, size);
    } else if (size < 65536) {
        relu_cpu_mt(in, out, size);
    } else {
        // Simuliere Batch-Dispatch an Metal
        std::vector<const float*> batch_in = {in};
        std::vector<float*> batch_out = {out};
        relu_metal_batch(batch_in, batch_out, size);
    }
}

} // namespace relu_dispatch

// --- TESTS ---

TEST(AdaptiveDispatch, SmallArrayUsesCPUSingleThread) {
    std::vector<float> in(128), out(128);
    std::iota(in.begin(), in.end(), -64.0f);
    relu_dispatch::relu_adaptive(in.data(), out.data(), in.size());
    for (size_t i = 0; i < in.size(); ++i)
        ASSERT_FLOAT_EQ(out[i], std::max(in[i], 0.0f));
}

TEST(AdaptiveDispatch, MediumArrayUsesCPUMultiThread) {
    std::vector<float> in(8192), out(8192);
    std::iota(in.begin(), in.end(), -4096.0f);
    relu_dispatch::relu_adaptive(in.data(), out.data(), in.size());
    for (size_t i = 0; i < in.size(); ++i)
        ASSERT_FLOAT_EQ(out[i], std::max(in[i], 0.0f));
}

TEST(AdaptiveDispatch, LargeArrayUsesMetalBatch) {
    std::vector<float> in(131072), out(131072);
    std::iota(in.begin(), in.end(), -65536.0f);
    relu_dispatch::relu_adaptive(in.data(), out.data(), in.size());
    for (size_t i = 0; i < in.size(); ++i)
        ASSERT_FLOAT_EQ(out[i], std::max(in[i], 0.0f));
}

TEST(AdaptiveDispatch, BatchedDispatchCorrectness) {
    int batch = 16;
    int size = 32768;
    std::vector<std::vector<float>> ins(batch, std::vector<float>(size));
    std::vector<std::vector<float>> outs(batch, std::vector<float>(size));
    for (int b = 0; b < batch; ++b)
        std::iota(ins[b].begin(), ins[b].end(), -size/2.0f + b);
    std::vector<const float*> ptr_in(batch);
    std::vector<float*> ptr_out(batch);
    for (int b = 0; b < batch; ++b) {
        ptr_in[b] = ins[b].data();
        ptr_out[b] = outs[b].data();
    }
    ASSERT_TRUE(relu_dispatch::relu_metal_batch(ptr_in, ptr_out, size));
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < size; ++i)
            ASSERT_FLOAT_EQ(outs[b][i], std::max(ins[b][i], 0.0f));
}
