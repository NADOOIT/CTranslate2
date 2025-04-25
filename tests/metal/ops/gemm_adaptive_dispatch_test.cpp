#include <gtest/gtest.h>
#include <vector>
#include <thread>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace gemm_dispatch {

void gemm_cpu(const float* a, const float* b, float* c, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l)
                sum += a[i * k + l] * b[l * n + j];
            c[i * n + j] = sum;
        }
}

void gemm_cpu_mt(const float* a, const float* b, float* c, int m, int n, int k, int num_threads = 0) {
    if (num_threads <= 0)
        num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk = (m + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int row_start = t * chunk;
        int row_end = std::min(row_start + chunk, m);
        threads.emplace_back([=]() {
            for (int i = row_start; i < row_end; ++i)
                for (int j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (int l = 0; l < k; ++l)
                        sum += a[i * k + l] * b[l * n + j];
                    c[i * n + j] = sum;
                }
        });
    }
    for (auto& th : threads) th.join();
}

extern "C" bool gemm_metal_batch(const std::vector<const float*>& a, const std::vector<const float*>& b, std::vector<float*>& c, int m, int n, int k);

void gemm_adaptive(const float* a, const float* b, float* c, int m, int n, int k) {
    if (m * n < 4096) {
        gemm_cpu(a, b, c, m, n, k);
    } else if (m * n < 65536) {
        gemm_cpu_mt(a, b, c, m, n, k);
    } else {
        std::vector<const float*> batch_a = {a};
        std::vector<const float*> batch_b = {b};
        std::vector<float*> batch_c = {c};
        gemm_metal_batch(batch_a, batch_b, batch_c, m, n, k);
    }
}

}

TEST(GEMMAdaptiveDispatch, SmallMatrixCPUSingle) {
    int m = 8, n = 8, k = 8;
    std::vector<float> a(m*k), b(k*n), c(m*n), ref(m*n);
    std::iota(a.begin(), a.end(), 1.0f);
    std::iota(b.begin(), b.end(), -32.0f);
    gemm_dispatch::gemm_adaptive(a.data(), b.data(), c.data(), m, n, k);
    gemm_dispatch::gemm_cpu(a.data(), b.data(), ref.data(), m, n, k);
    for (int i = 0; i < m*n; ++i)
        ASSERT_NEAR(c[i], ref[i], 1e-3f);
}

TEST(GEMMAdaptiveDispatch, MediumMatrixCPUMulti) {
    int m = 128, n = 128, k = 128;
    std::vector<float> a(m*k), b(k*n), c(m*n), ref(m*n);
    std::iota(a.begin(), a.end(), 1.0f);
    std::iota(b.begin(), b.end(), -32.0f);
    gemm_dispatch::gemm_adaptive(a.data(), b.data(), c.data(), m, n, k);
    gemm_dispatch::gemm_cpu(a.data(), b.data(), ref.data(), m, n, k);
    for (int i = 0; i < m*n; ++i)
        ASSERT_NEAR(c[i], ref[i], 1e-3f);
}

TEST(GEMMAdaptiveDispatch, LargeMatrixMetalBatch) {
    int m = 512, n = 512, k = 512;
    std::vector<float> a(m*k), b(k*n), c(m*n), ref(m*n);
    std::iota(a.begin(), a.end(), 1.0f);
    std::iota(b.begin(), b.end(), -32.0f);
    gemm_dispatch::gemm_adaptive(a.data(), b.data(), c.data(), m, n, k);
    gemm_dispatch::gemm_cpu(a.data(), b.data(), ref.data(), m, n, k);
    for (int i = 0; i < m*n; ++i)
        ASSERT_NEAR(c[i], ref[i], 1e-2f);
}

TEST(GEMMAdaptiveDispatch, BatchedDispatchCorrectness) {
    int batch = 8, m = 64, n = 64, k = 64;
    std::vector<std::vector<float>> as(batch, std::vector<float>(m*k));
    std::vector<std::vector<float>> bs(batch, std::vector<float>(k*n));
    std::vector<std::vector<float>> cs(batch, std::vector<float>(m*n));
    std::vector<std::vector<float>> refs(batch, std::vector<float>(m*n));
    for (int b = 0; b < batch; ++b) {
        std::iota(as[b].begin(), as[b].end(), 1.0f + b);
        std::iota(bs[b].begin(), bs[b].end(), -32.0f + b);
    }
    std::vector<const float*> ptr_a(batch), ptr_b(batch);
    std::vector<float*> ptr_c(batch);
    for (int b = 0; b < batch; ++b) {
        ptr_a[b] = as[b].data();
        ptr_b[b] = bs[b].data();
        ptr_c[b] = cs[b].data();
        gemm_dispatch::gemm_cpu(as[b].data(), bs[b].data(), refs[b].data(), m, n, k);
    }
    ASSERT_TRUE(gemm_dispatch::gemm_metal_batch(ptr_a, ptr_b, ptr_c, m, n, k));
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < m*n; ++i)
            ASSERT_NEAR(cs[b][i], refs[b][i], 1e-2f);
}
