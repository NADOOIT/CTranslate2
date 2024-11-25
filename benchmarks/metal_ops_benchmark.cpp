#include "metal/metal_device.h"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <memory>
#include <string>
#include <iomanip>
#include <sstream>

namespace ctranslate2 {
  namespace metal {
    namespace benchmarks {

      // Helper function to format bandwidth in human-readable form
      std::string format_bandwidth(double bytes_per_second) {
        const char* units[] = {"B/s", "KB/s", "MB/s", "GB/s", "TB/s"};
        int unit = 0;
        while (bytes_per_second >= 1024.0 && unit < 4) {
          bytes_per_second /= 1024.0;
          unit++;
        }
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << bytes_per_second << " " << units[unit];
        return ss.str();
      }

      template<typename T>
      class MetalBenchmark : public benchmark::Fixture {
      protected:
        void SetUp(const benchmark::State& state) override {
          device = std::make_unique<MetalDevice>();
          size = state.range(0);
          
          // Generate random data
          input_data = generate_random_data<T>(size);
          residual_data = generate_random_data<T>(size);
          
          // Allocate device memory
          input_buffer = device->allocate<T>(size);
          residual_buffer = device->allocate<T>(size);
          output_buffer = device->allocate<T>(size);
          temp_buffer = device->allocate<T>(size);
          
          // Copy data to device
          device->copy_to_device<T>(input_data.data(), input_buffer, size);
          device->copy_to_device<T>(residual_data.data(), residual_buffer, size);
        }

        void TearDown(const benchmark::State& state) override {
          device->free(input_buffer);
          device->free(residual_buffer);
          device->free(output_buffer);
          device->free(temp_buffer);
        }

        template <typename U>
        std::vector<U> generate_random_data(size_t size, U min = U(-1), U max = U(1)) {
          std::vector<U> data(size);
          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_real_distribution<float> dis(min, max);
          
          for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<U>(dis(gen));
          }
          
          return data;
        }

        void ReportBandwidth(benchmark::State& state) {
          double time = state.iterations() * state.duration_cpu().count() * 1e-9;  // Convert to seconds
          double bytes = state.bytes_processed();
          double bandwidth = bytes / time;
          state.counters["Bandwidth"] = benchmark::Counter(bandwidth,
                                                         benchmark::Counter::kDefaults,
                                                         benchmark::Counter::OneK::kIs1024);
          state.SetLabel(format_bandwidth(bandwidth));
        }

        std::unique_ptr<MetalDevice> device;
        size_t size;
        std::vector<T> input_data;
        std::vector<T> residual_data;
        void* input_buffer;
        void* residual_buffer;
        void* output_buffer;
        void* temp_buffer;
      };

      // Benchmark GELU + Add (Fused)
      template<typename T>
      void BM_GeluAddFused(benchmark::State& state) {
        MetalBenchmark<T> bm;
        bm.SetUp(state);

        for (auto _ : state) {
          bm.device->gelu_add<T>(
            static_cast<T*>(bm.output_buffer),
            static_cast<const T*>(bm.input_buffer),
            static_cast<const T*>(bm.residual_buffer),
            bm.size);
          
          benchmark::DoNotOptimize(bm.output_buffer);
          benchmark::ClobberMemory();
        }
        
        state.SetItemsProcessed(state.iterations() * bm.size);
        state.SetBytesProcessed(state.iterations() * bm.size * sizeof(T) * 3);
        bm.ReportBandwidth(state);
        
        bm.TearDown(state);
      }

      // Benchmark GELU + Add (Unfused)
      template<typename T>
      void BM_GeluAddUnfused(benchmark::State& state) {
        MetalBenchmark<T> bm;
        bm.SetUp(state);

        for (auto _ : state) {
          bm.device->gelu<T>(
            static_cast<T*>(bm.temp_buffer),
            static_cast<const T*>(bm.input_buffer),
            bm.size);
          
          bm.device->add<T>(
            static_cast<T*>(bm.output_buffer),
            static_cast<const T*>(bm.temp_buffer),
            static_cast<const T*>(bm.residual_buffer),
            bm.size);
          
          benchmark::DoNotOptimize(bm.output_buffer);
          benchmark::ClobberMemory();
        }
        
        state.SetItemsProcessed(state.iterations() * bm.size);
        state.SetBytesProcessed(state.iterations() * bm.size * sizeof(T) * 4);
        bm.ReportBandwidth(state);
        
        bm.TearDown(state);
      }

      // Benchmark ReLU + Add (Fused)
      template<typename T>
      void BM_ReluAddFused(benchmark::State& state) {
        MetalBenchmark<T> bm;
        bm.SetUp(state);

        for (auto _ : state) {
          bm.device->relu_add<T>(
            static_cast<T*>(bm.output_buffer),
            static_cast<const T*>(bm.input_buffer),
            static_cast<const T*>(bm.residual_buffer),
            bm.size);
          
          benchmark::DoNotOptimize(bm.output_buffer);
          benchmark::ClobberMemory();
        }
        
        state.SetItemsProcessed(state.iterations() * bm.size);
        state.SetBytesProcessed(state.iterations() * bm.size * sizeof(T) * 3);
        bm.ReportBandwidth(state);
        
        bm.TearDown(state);
      }

      // Benchmark ReLU + Add (Unfused)
      template<typename T>
      void BM_ReluAddUnfused(benchmark::State& state) {
        MetalBenchmark<T> bm;
        bm.SetUp(state);

        for (auto _ : state) {
          bm.device->relu<T>(
            static_cast<T*>(bm.temp_buffer),
            static_cast<const T*>(bm.input_buffer),
            bm.size);
          
          bm.device->add<T>(
            static_cast<T*>(bm.output_buffer),
            static_cast<const T*>(bm.temp_buffer),
            static_cast<const T*>(bm.residual_buffer),
            bm.size);
          
          benchmark::DoNotOptimize(bm.output_buffer);
          benchmark::ClobberMemory();
        }
        
        state.SetItemsProcessed(state.iterations() * bm.size);
        state.SetBytesProcessed(state.iterations() * bm.size * sizeof(T) * 4);
        bm.ReportBandwidth(state);
        
        bm.TearDown(state);
      }

      // Register float32 benchmarks
      BENCHMARK_TEMPLATE(BM_GeluAddFused, float)
        ->RangeMultiplier(2)
        ->Range(1<<10, 1<<20)
        ->Unit(benchmark::kMicrosecond)
        ->UseRealTime()
        ->Name("Float32_GeluAdd_Fused");

      BENCHMARK_TEMPLATE(BM_GeluAddUnfused, float)
        ->RangeMultiplier(2)
        ->Range(1<<10, 1<<20)
        ->Unit(benchmark::kMicrosecond)
        ->UseRealTime()
        ->Name("Float32_GeluAdd_Unfused");

      BENCHMARK_TEMPLATE(BM_ReluAddFused, float)
        ->RangeMultiplier(2)
        ->Range(1<<10, 1<<20)
        ->Unit(benchmark::kMicrosecond)
        ->UseRealTime()
        ->Name("Float32_ReluAdd_Fused");

      BENCHMARK_TEMPLATE(BM_ReluAddUnfused, float)
        ->RangeMultiplier(2)
        ->Range(1<<10, 1<<20)
        ->Unit(benchmark::kMicrosecond)
        ->UseRealTime()
        ->Name("Float32_ReluAdd_Unfused");

      // Register float16 benchmarks
      BENCHMARK_TEMPLATE(BM_GeluAddFused, __fp16)
        ->RangeMultiplier(2)
        ->Range(1<<10, 1<<20)
        ->Unit(benchmark::kMicrosecond)
        ->UseRealTime()
        ->Name("Float16_GeluAdd_Fused");

      BENCHMARK_TEMPLATE(BM_GeluAddUnfused, __fp16)
        ->RangeMultiplier(2)
        ->Range(1<<10, 1<<20)
        ->Unit(benchmark::kMicrosecond)
        ->UseRealTime()
        ->Name("Float16_GeluAdd_Unfused");

      BENCHMARK_TEMPLATE(BM_ReluAddFused, __fp16)
        ->RangeMultiplier(2)
        ->Range(1<<10, 1<<20)
        ->Unit(benchmark::kMicrosecond)
        ->UseRealTime()
        ->Name("Float16_ReluAdd_Fused");

      BENCHMARK_TEMPLATE(BM_ReluAddUnfused, __fp16)
        ->RangeMultiplier(2)
        ->Range(1<<10, 1<<20)
        ->Unit(benchmark::kMicrosecond)
        ->UseRealTime()
        ->Name("Float16_ReluAdd_Unfused");

    }  // namespace benchmarks
  }  // namespace metal
}  // namespace ctranslate2

BENCHMARK_MAIN();
