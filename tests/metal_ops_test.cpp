#include "metal/metal_device.h"
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

namespace ctranslate2 {
  namespace metal {
    namespace tests {

      class MetalDeviceTest : public ::testing::Test {
      protected:
        void SetUp() override {
          device = std::make_unique<MetalDevice>();
        }

        template <typename T>
        std::vector<T> generate_random_data(size_t size, T min = T(-1), T max = T(1)) {
          std::vector<T> data(size);
          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_real_distribution<float> dis(min, max);
          
          for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<T>(dis(gen));
          }
          
          return data;
        }

        template <typename T>
        T gelu_reference(T x) {
          const T sqrt_2_over_pi = T(0.7978845608028654);
          const T coef = T(0.044715);
          return x * T(0.5) * (T(1.0) + std::tanh(sqrt_2_over_pi * (x + coef * x * x * x)));
        }

        template <typename T>
        T relu_reference(T x) {
          return x > T(0) ? x : T(0);
        }

        template <typename T>
        void verify_close(const std::vector<T>& actual,
                         const std::vector<T>& expected,
                         T tolerance = T(1e-5)) {
          ASSERT_EQ(actual.size(), expected.size());
          for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], tolerance)
              << "Values differ at index " << i;
          }
        }

        std::unique_ptr<MetalDevice> device;
      };

      TEST_F(MetalDeviceTest, TestGeluAdd) {
        const size_t size = 1024;
        auto input = generate_random_data<float>(size);
        auto residual = generate_random_data<float>(size);
        std::vector<float> output(size);
        std::vector<float> expected(size);

        // Compute reference
        for (size_t i = 0; i < size; ++i) {
          expected[i] = gelu_reference(input[i]) + residual[i];
        }

        // Compute using Metal
        void* input_buffer = device->allocate<float>(size);
        void* residual_buffer = device->allocate<float>(size);
        void* output_buffer = device->allocate<float>(size);

        device->copy_to_device<float>(input.data(), input_buffer, size);
        device->copy_to_device<float>(residual.data(), residual_buffer, size);

        device->gelu_add<float>(
          static_cast<float*>(output_buffer),
          static_cast<const float*>(input_buffer),
          static_cast<const float*>(residual_buffer),
          size);

        device->copy_from_device<float>(output_buffer, output.data(), size);

        verify_close(output, expected);

        device->free(input_buffer);
        device->free(residual_buffer);
        device->free(output_buffer);
      }

      TEST_F(MetalDeviceTest, TestReluAdd) {
        const size_t size = 1024;
        auto input = generate_random_data<float>(size);
        auto residual = generate_random_data<float>(size);
        std::vector<float> output(size);
        std::vector<float> expected(size);

        // Compute reference
        for (size_t i = 0; i < size; ++i) {
          expected[i] = relu_reference(input[i]) + residual[i];
        }

        // Compute using Metal
        void* input_buffer = device->allocate<float>(size);
        void* residual_buffer = device->allocate<float>(size);
        void* output_buffer = device->allocate<float>(size);

        device->copy_to_device<float>(input.data(), input_buffer, size);
        device->copy_to_device<float>(residual.data(), residual_buffer, size);

        device->relu_add<float>(
          static_cast<float*>(output_buffer),
          static_cast<const float*>(input_buffer),
          static_cast<const float*>(residual_buffer),
          size);

        device->copy_from_device<float>(output_buffer, output.data(), size);

        verify_close(output, expected);

        device->free(input_buffer);
        device->free(residual_buffer);
        device->free(output_buffer);
      }

      TEST_F(MetalDeviceTest, TestGeluAddHalf) {
        const size_t size = 1024;
        auto input = generate_random_data<float>(size);
        auto residual = generate_random_data<float>(size);
        std::vector<float> output(size);
        std::vector<float> expected(size);

        // Convert to half precision
        std::vector<__fp16> input_half(size);
        std::vector<__fp16> residual_half(size);
        std::vector<__fp16> output_half(size);

        for (size_t i = 0; i < size; ++i) {
          input_half[i] = static_cast<__fp16>(input[i]);
          residual_half[i] = static_cast<__fp16>(residual[i]);
          expected[i] = gelu_reference(input[i]) + residual[i];
        }

        // Compute using Metal
        void* input_buffer = device->allocate<__fp16>(size);
        void* residual_buffer = device->allocate<__fp16>(size);
        void* output_buffer = device->allocate<__fp16>(size);

        device->copy_to_device<__fp16>(input_half.data(), input_buffer, size);
        device->copy_to_device<__fp16>(residual_half.data(), residual_buffer, size);

        device->gelu_add<__fp16>(
          static_cast<__fp16*>(output_buffer),
          static_cast<const __fp16*>(input_buffer),
          static_cast<const __fp16*>(residual_buffer),
          size);

        device->copy_from_device<__fp16>(output_buffer, output_half.data(), size);

        // Convert back to float for comparison
        for (size_t i = 0; i < size; ++i) {
          output[i] = static_cast<float>(output_half[i]);
        }

        verify_close(output, expected, 1e-2f);  // Larger tolerance for half precision

        device->free(input_buffer);
        device->free(residual_buffer);
        device->free(output_buffer);
      }

      TEST_F(MetalDeviceTest, TestReluAddHalf) {
        const size_t size = 1024;
        auto input = generate_random_data<float>(size);
        auto residual = generate_random_data<float>(size);
        std::vector<float> output(size);
        std::vector<float> expected(size);

        // Convert to half precision
        std::vector<__fp16> input_half(size);
        std::vector<__fp16> residual_half(size);
        std::vector<__fp16> output_half(size);

        for (size_t i = 0; i < size; ++i) {
          input_half[i] = static_cast<__fp16>(input[i]);
          residual_half[i] = static_cast<__fp16>(residual[i]);
          expected[i] = relu_reference(input[i]) + residual[i];
        }

        // Compute using Metal
        void* input_buffer = device->allocate<__fp16>(size);
        void* residual_buffer = device->allocate<__fp16>(size);
        void* output_buffer = device->allocate<__fp16>(size);

        device->copy_to_device<__fp16>(input_half.data(), input_buffer, size);
        device->copy_to_device<__fp16>(residual_half.data(), residual_buffer, size);

        device->relu_add<__fp16>(
          static_cast<__fp16*>(output_buffer),
          static_cast<const __fp16*>(input_buffer),
          static_cast<const __fp16*>(residual_buffer),
          size);

        device->copy_from_device<__fp16>(output_buffer, output_half.data(), size);

        // Convert back to float for comparison
        for (size_t i = 0; i < size; ++i) {
          output[i] = static_cast<float>(output_half[i]);
        }

        verify_close(output, expected, 1e-2f);  // Larger tolerance for half precision

        device->free(input_buffer);
        device->free(residual_buffer);
        device->free(output_buffer);
      }

    }  // namespace tests
  }  // namespace metal
}  // namespace ctranslate2
