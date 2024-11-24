#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#include "metal/metal_device.h"
#include "metal/metal_allocator.h"
#include "metal/metal_kernels.h"

using namespace ctranslate2;

class MetalBenchmark : public benchmark::Fixture {
protected:
  void SetUp(const benchmark::State& state) override {
    device = std::make_unique<metal::MetalDevice>(0);
    allocator = std::make_unique<metal::MetalAllocator>(*device);
  }

  void TearDown(const benchmark::State& state) override {
    device.reset();
    allocator.reset();
  }

  template <typename T>
  void* toDevice(const std::vector<T>& host_data) {
    return allocator->host_to_device(host_data.data(), host_data.size() * sizeof(T));
  }

  template <typename T>
  std::vector<T> generateRandomData(size_t size) {
    std::vector<T> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-1.0, 1.0);
    for (size_t i = 0; i < size; ++i) {
      data[i] = dis(gen);
    }
    return data;
  }

  std::unique_ptr<metal::MetalDevice> device;
  std::unique_ptr<metal::MetalAllocator> allocator;
};

BENCHMARK_DEFINE_F(MetalBenchmark, Add)(benchmark::State& state) {
  const size_t size = state.range(0);
  auto a = generateRandomData<float>(size);
  auto b = generateRandomData<float>(size);

  void* d_a = toDevice(a);
  void* d_b = toDevice(b);
  void* d_c = allocator->allocate(size * sizeof(float));

  for (auto _ : state) {
    metal::add(*device,
              static_cast<const float*>(d_a),
              static_cast<const float*>(d_b),
              static_cast<float*>(d_c),
              size);
    device->synchronize();
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * size * sizeof(float) * 3);
  state.SetItemsProcessed(int64_t(state.iterations()) * size);

  allocator->free(d_a);
  allocator->free(d_b);
  allocator->free(d_c);
}

BENCHMARK_DEFINE_F(MetalBenchmark, GEMM)(benchmark::State& state) {
  const size_t m = state.range(0);
  const size_t n = state.range(0);
  const size_t k = state.range(0);

  auto a = generateRandomData<float>(m * k);
  auto b = generateRandomData<float>(k * n);

  void* d_a = toDevice(a);
  void* d_b = toDevice(b);
  void* d_c = allocator->allocate(m * n * sizeof(float));

  for (auto _ : state) {
    metal::gemm(*device,
               false,  // a_trans
               false,  // b_trans
               m, n, k,
               1.0f,   // alpha
               static_cast<const float*>(d_a),
               static_cast<const float*>(d_b),
               0.0f,   // beta
               static_cast<float*>(d_c));
    device->synchronize();
  }

  // Each GEMM operation performs 2*M*N*K floating point operations
  state.SetBytesProcessed(int64_t(state.iterations()) * (m * k + k * n + m * n) * sizeof(float));
  state.SetItemsProcessed(int64_t(state.iterations()) * 2 * m * n * k);

  allocator->free(d_a);
  allocator->free(d_b);
  allocator->free(d_c);
}

BENCHMARK_DEFINE_F(MetalBenchmark, LayerNorm)(benchmark::State& state) {
  const size_t batch_size = 32;
  const size_t hidden_size = state.range(0);

  auto input = generateRandomData<float>(batch_size * hidden_size);
  auto gamma = generateRandomData<float>(hidden_size);
  auto beta = generateRandomData<float>(hidden_size);

  void* d_input = toDevice(input);
  void* d_gamma = toDevice(gamma);
  void* d_beta = toDevice(beta);
  void* d_output = allocator->allocate(batch_size * hidden_size * sizeof(float));

  for (auto _ : state) {
    metal::layer_norm(*device,
                    static_cast<const float*>(d_input),
                    static_cast<const float*>(d_gamma),
                    static_cast<const float*>(d_beta),
                    static_cast<float*>(d_output),
                    batch_size,
                    hidden_size);
    device->synchronize();
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * batch_size * hidden_size * sizeof(float) * 4);
  state.SetItemsProcessed(int64_t(state.iterations()) * batch_size * hidden_size);

  allocator->free(d_input);
  allocator->free(d_gamma);
  allocator->free(d_beta);
  allocator->free(d_output);
}

BENCHMARK_DEFINE_F(MetalBenchmark, Softmax)(benchmark::State& state) {
  const size_t batch_size = 32;
  const size_t depth = state.range(0);

  auto input = generateRandomData<float>(batch_size * depth);

  void* d_input = toDevice(input);
  void* d_output = allocator->allocate(batch_size * depth * sizeof(float));

  for (auto _ : state) {
    metal::softmax(*device,
                  static_cast<const float*>(d_input),
                  static_cast<float*>(d_output),
                  batch_size,
                  depth);
    device->synchronize();
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * batch_size * depth * sizeof(float) * 2);
  state.SetItemsProcessed(int64_t(state.iterations()) * batch_size * depth);

  allocator->free(d_input);
  allocator->free(d_output);
}

// Register benchmarks with different input sizes
BENCHMARK_REGISTER_F(MetalBenchmark, Add)
    ->RangeMultiplier(2)
    ->Range(1<<10, 1<<20)  // Test from 1K to 1M elements
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(MetalBenchmark, GEMM)
    ->RangeMultiplier(2)
    ->Range(128, 2048)     // Test matrix sizes from 128x128 to 2048x2048
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(MetalBenchmark, LayerNorm)
    ->RangeMultiplier(2)
    ->Range(256, 4096)     // Test hidden sizes from 256 to 4096
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(MetalBenchmark, Softmax)
    ->RangeMultiplier(2)
    ->Range(256, 4096)     // Test depth sizes from 256 to 4096
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
