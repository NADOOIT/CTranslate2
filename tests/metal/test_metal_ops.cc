#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "metal/metal_device.h"
#include "metal/metal_allocator.h"
#include "metal/metal_kernels.h"

using namespace ctranslate2;

class MetalTest : public ::testing::Test {
protected:
  void SetUp() override {
    device = std::make_unique<metal::MetalDevice>(0);
    allocator = std::make_unique<metal::MetalAllocator>(*device);
  }

  template <typename T>
  void* toDevice(const std::vector<T>& host_data) {
    return allocator->host_to_device(host_data.data(), host_data.size() * sizeof(T));
  }

  template <typename T>
  std::vector<T> toHost(void* device_data, size_t size) {
    std::vector<T> host_data(size);
    allocator->device_to_host(device_data, host_data.data(), size * sizeof(T));
    return host_data;
  }

  std::unique_ptr<metal::MetalDevice> device;
  std::unique_ptr<metal::MetalAllocator> allocator;
};

TEST_F(MetalTest, Add) {
  const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> b = {2.0f, 3.0f, 4.0f, 5.0f};
  const std::vector<float> expected = {3.0f, 5.0f, 7.0f, 9.0f};

  void* d_a = toDevice(a);
  void* d_b = toDevice(b);
  void* d_c = allocator->allocate(a.size() * sizeof(float));

  metal::add(*device, 
            static_cast<const float*>(d_a),
            static_cast<const float*>(d_b),
            static_cast<float*>(d_c),
            a.size());

  auto result = toHost<float>(d_c, a.size());
  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_FLOAT_EQ(result[i], expected[i]);
  }

  allocator->free(d_a);
  allocator->free(d_b);
  allocator->free(d_c);
}

TEST_F(MetalTest, Multiply) {
  const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> b = {2.0f, 3.0f, 4.0f, 5.0f};
  const std::vector<float> expected = {2.0f, 6.0f, 12.0f, 20.0f};

  void* d_a = toDevice(a);
  void* d_b = toDevice(b);
  void* d_c = allocator->allocate(a.size() * sizeof(float));

  metal::multiply(*device,
                static_cast<const float*>(d_a),
                static_cast<const float*>(d_b),
                static_cast<float*>(d_c),
                a.size());

  auto result = toHost<float>(d_c, a.size());
  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_FLOAT_EQ(result[i], expected[i]);
  }

  allocator->free(d_a);
  allocator->free(d_b);
  allocator->free(d_c);
}

TEST_F(MetalTest, ReLU) {
  const std::vector<float> x = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  const std::vector<float> expected = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};

  void* d_x = toDevice(x);
  void* d_y = allocator->allocate(x.size() * sizeof(float));

  metal::relu(*device,
            static_cast<const float*>(d_x),
            static_cast<float*>(d_y),
            x.size());

  auto result = toHost<float>(d_y, x.size());
  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_FLOAT_EQ(result[i], expected[i]);
  }

  allocator->free(d_x);
  allocator->free(d_y);
}

TEST_F(MetalTest, Softmax) {
  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  const size_t batch_size = 1;
  const size_t depth = input.size();

  void* d_input = toDevice(input);
  void* d_output = allocator->allocate(input.size() * sizeof(float));

  metal::softmax(*device,
                static_cast<const float*>(d_input),
                static_cast<float*>(d_output),
                batch_size,
                depth);

  auto result = toHost<float>(d_output, input.size());
  
  // Verify properties of softmax
  ASSERT_EQ(result.size(), input.size());
  
  // Sum should be close to 1
  float sum = 0.0f;
  for (float val : result) {
    EXPECT_GE(val, 0.0f);
    EXPECT_LE(val, 1.0f);
    sum += val;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-6);

  // Values should be monotonically increasing
  for (size_t i = 1; i < result.size(); ++i) {
    EXPECT_GT(result[i], result[i-1]);
  }

  allocator->free(d_input);
  allocator->free(d_output);
}

TEST_F(MetalTest, LayerNorm) {
  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> gamma = {1.0f, 1.0f, 1.0f, 1.0f};
  const std::vector<float> beta = {0.0f, 0.0f, 0.0f, 0.0f};
  const size_t batch_size = 1;
  const size_t hidden_size = input.size();

  void* d_input = toDevice(input);
  void* d_gamma = toDevice(gamma);
  void* d_beta = toDevice(beta);
  void* d_output = allocator->allocate(input.size() * sizeof(float));

  metal::layer_norm(*device,
                  static_cast<const float*>(d_input),
                  static_cast<const float*>(d_gamma),
                  static_cast<const float*>(d_beta),
                  static_cast<float*>(d_output),
                  batch_size,
                  hidden_size);

  auto result = toHost<float>(d_output, input.size());
  
  // Verify properties of layer normalization
  ASSERT_EQ(result.size(), input.size());
  
  // Mean should be close to 0
  float mean = 0.0f;
  for (float val : result) {
    mean += val;
  }
  mean /= result.size();
  EXPECT_NEAR(mean, 0.0f, 1e-6);

  // Variance should be close to 1
  float variance = 0.0f;
  for (float val : result) {
    variance += (val - mean) * (val - mean);
  }
  variance /= result.size();
  EXPECT_NEAR(variance, 1.0f, 1e-6);

  allocator->free(d_input);
  allocator->free(d_gamma);
  allocator->free(d_beta);
  allocator->free(d_output);
}

TEST_F(MetalTest, GEMM) {
  // 2x2 matrices
  const std::vector<float> a = {1.0f, 2.0f,
                               3.0f, 4.0f};
  const std::vector<float> b = {5.0f, 6.0f,
                               7.0f, 8.0f};
  const std::vector<float> expected = {19.0f, 22.0f,
                                     43.0f, 50.0f};
  const size_t m = 2;
  const size_t n = 2;
  const size_t k = 2;

  void* d_a = toDevice(a);
  void* d_b = toDevice(b);
  void* d_c = allocator->allocate(m * n * sizeof(float));

  metal::gemm(*device,
            false,  // a_trans
            false,  // b_trans
            m, n, k,
            1.0f,   // alpha
            static_cast<const float*>(d_a),
            static_cast<const float*>(d_b),
            0.0f,   // beta
            static_cast<float*>(d_c));

  auto result = toHost<float>(d_c, m * n);
  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }

  allocator->free(d_a);
  allocator->free(d_b);
  allocator->free(d_c);
}
