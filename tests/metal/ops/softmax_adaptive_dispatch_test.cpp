#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

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

extern "C" bool softmax_metal_batch(const std::vector<const float*>& ins, std::vector<float*>& outs, int size);

void softmax_adaptive(const float* in, float* out, int size) {
    if (size < 4096) {
        softmax_cpu(in, out, size);
    } else if (size < 65536) {
        softmax_cpu_mt(in, out, size);
    } else {
        std::vector<const float*> batch_in = {in};
        std::vector<float*> batch_out = {out};
        softmax_metal_batch(batch_in, batch_out, size);
    }
}

} // namespace softmax_dispatch

TEST(SoftmaxAdaptiveDispatch, SmallArrayCPUSingle) {
    int size = 128;
    std::vector<float> in(size), out(size), ref(size);
    std::iota(in.begin(), in.end(), -64.0f);
    softmax_dispatch::softmax_adaptive(in.data(), out.data(), size);
    softmax_dispatch::softmax_cpu(in.data(), ref.data(), size);
    for (int i = 0; i < size; ++i)
        ASSERT_NEAR(out[i], ref[i], 1e-5f);
}

TEST(SoftmaxAdaptiveDispatch, MediumArrayCPUMulti) {
    int size = 8192;
    std::vector<float> in(size), out(size), ref(size);
    std::iota(in.begin(), in.end(), -4096.0f);
    softmax_dispatch::softmax_adaptive(in.data(), out.data(), size);
    softmax_dispatch::softmax_cpu(in.data(), ref.data(), size);
    for (int i = 0; i < size; ++i)
        ASSERT_NEAR(out[i], ref[i], 1e-5f);
}

TEST(SoftmaxAdaptiveDispatch, LargeArrayMetalBatch) {
    int size = 131072;
    std::vector<float> in(size), out(size), ref(size);
    std::iota(in.begin(), in.end(), -65536.0f);
    softmax_dispatch::softmax_adaptive(in.data(), out.data(), size);
    softmax_dispatch::softmax_cpu(in.data(), ref.data(), size);
    for (int i = 0; i < size; ++i)
        ASSERT_NEAR(out[i], ref[i], 1e-4f);
}

TEST(SoftmaxAdaptiveDispatch, BatchedDispatchCorrectness) {
    int batch = 8, size = 32768;
    std::vector<std::vector<float>> ins(batch, std::vector<float>(size));
    std::vector<std::vector<float>> outs(batch, std::vector<float>(size));
    std::vector<std::vector<float>> refs(batch, std::vector<float>(size));
    for (int b = 0; b < batch; ++b)
        std::iota(ins[b].begin(), ins[b].end(), -size/2.0f + b);
    std::vector<const float*> ptr_in(batch);
    std::vector<float*> ptr_out(batch);
    for (int b = 0; b < batch; ++b) {
        ptr_in[b] = ins[b].data();
        ptr_out[b] = outs[b].data();
        softmax_dispatch::softmax_cpu(ins[b].data(), refs[b].data(), size);
    }
    ASSERT_TRUE(softmax_dispatch::softmax_metal_batch(ptr_in, ptr_out, size));
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < size; ++i)
            ASSERT_NEAR(outs[b][i], refs[b][i], 1e-4f);
}
