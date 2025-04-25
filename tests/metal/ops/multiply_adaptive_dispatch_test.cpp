#include <gtest/gtest.h>
#include <vector>
#include <thread>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace multiply_dispatch {

void multiply_cpu(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; ++i)
        c[i] = a[i] * b[i];
}

void multiply_cpu_mt(const float* a, const float* b, float* c, int size, int num_threads = 0) {
    if (num_threads <= 0)
        num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk = (size + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk;
        int end = std::min(start + chunk, size);
        threads.emplace_back([=]() {
            for (int i = start; i < end; ++i)
                c[i] = a[i] * b[i];
        });
    }
    for (auto& th : threads) th.join();
}

extern "C" bool multiply_metal_batch(const std::vector<const float*>& a, const std::vector<const float*>& b, std::vector<float*>& c, int size);

void multiply_adaptive(const float* a, const float* b, float* c, int size) {
    if (size < 4096) {
        multiply_cpu(a, b, c, size);
    } else if (size < 65536) {
        multiply_cpu_mt(a, b, c, size);
    } else {
        std::vector<const float*> batch_a = {a};
        std::vector<const float*> batch_b = {b};
        std::vector<float*> batch_c = {c};
        multiply_metal_batch(batch_a, batch_b, batch_c, size);
    }
}

}

TEST(MultiplyAdaptiveDispatch, SmallArrayCPUSingle) {
    int size = 128;
    std::vector<float> a(size), b(size), c(size), ref(size);
    std::iota(a.begin(), a.end(), 1.0f);
    std::iota(b.begin(), b.end(), -64.0f);
    multiply_dispatch::multiply_adaptive(a.data(), b.data(), c.data(), size);
    multiply_dispatch::multiply_cpu(a.data(), b.data(), ref.data(), size);
    for (int i = 0; i < size; ++i)
        ASSERT_FLOAT_EQ(c[i], ref[i]);
}

TEST(MultiplyAdaptiveDispatch, MediumArrayCPUMulti) {
    int size = 8192;
    std::vector<float> a(size), b(size), c(size), ref(size);
    std::iota(a.begin(), a.end(), 1.0f);
    std::iota(b.begin(), b.end(), -4096.0f);
    multiply_dispatch::multiply_adaptive(a.data(), b.data(), c.data(), size);
    multiply_dispatch::multiply_cpu(a.data(), b.data(), ref.data(), size);
    for (int i = 0; i < size; ++i)
        ASSERT_FLOAT_EQ(c[i], ref[i]);
}

TEST(MultiplyAdaptiveDispatch, LargeArrayMetalBatch) {
    int size = 131072;
    std::vector<float> a(size), b(size), c(size), ref(size);
    std::iota(a.begin(), a.end(), 1.0f);
    std::iota(b.begin(), b.end(), -65536.0f);
    multiply_dispatch::multiply_adaptive(a.data(), b.data(), c.data(), size);
    multiply_dispatch::multiply_cpu(a.data(), b.data(), ref.data(), size);
    for (int i = 0; i < size; ++i)
        ASSERT_FLOAT_EQ(c[i], ref[i]);
}

TEST(MultiplyAdaptiveDispatch, BatchedDispatchCorrectness) {
    int batch = 16, size = 32768;
    std::vector<std::vector<float>> as(batch, std::vector<float>(size));
    std::vector<std::vector<float>> bs(batch, std::vector<float>(size));
    std::vector<std::vector<float>> cs(batch, std::vector<float>(size));
    std::vector<std::vector<float>> refs(batch, std::vector<float>(size));
    for (int b = 0; b < batch; ++b) {
        std::iota(as[b].begin(), as[b].end(), 1.0f + b);
        std::iota(bs[b].begin(), bs[b].end(), -size/2.0f + b);
    }
    std::vector<const float*> ptr_a(batch), ptr_b(batch);
    std::vector<float*> ptr_c(batch);
    for (int b = 0; b < batch; ++b) {
        ptr_a[b] = as[b].data();
        ptr_b[b] = bs[b].data();
        ptr_c[b] = cs[b].data();
        multiply_dispatch::multiply_cpu(as[b].data(), bs[b].data(), refs[b].data(), size);
    }
    ASSERT_TRUE(multiply_dispatch::multiply_metal_batch(ptr_a, ptr_b, ptr_c, size));
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < size; ++i)
            ASSERT_FLOAT_EQ(cs[b][i], refs[b][i]);
}
