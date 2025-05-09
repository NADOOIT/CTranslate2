\
#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include <vector>

#include "test_utils.h" // For test helpers like expect_storage_eq
#include "ctranslate2/primitives.h" // For primitives<Device::METAL>::attention
#include "ctranslate2/storage_view.h"
#include "device_dispatch.h"
#include "../../src/metal/utils.h"             // For has_metal, get_metal_device
#include "../../src/metal/metal_allocator.h" // For get_metal_allocator

// Assuming MetalTest fixture is accessible (might need to include the header from test_metal_ops.cc or redefine it)
// For now, let's redefine a minimal version. TODO: Refactor test fixtures later if needed.
class MetalAttentionTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!ctranslate2::metal::has_metal()) {
       GTEST_SKIP() << "Metal is not available on this system, skipping test.";
    }
    ctranslate2::metal::set_metal_device(0); // Set the default device
    _allocator = &ctranslate2::get_allocator(ctranslate2::Device::METAL); // Get allocator for METAL device
  }

  template <typename T>
  ctranslate2::StorageView create_metal_storage(const ctranslate2::Shape& shape, const std::vector<T>& data) {
      ctranslate2::StorageView metal_storage({shape}, static_cast<T>(0), ctranslate2::Device::METAL);
      ctranslate2::StorageView cpu_storage({shape}, data, ctranslate2::Device::CPU);
      metal_storage.copy_from(cpu_storage);
      return metal_storage;
  }

  template <typename T>
  std::vector<T> get_vector(const ctranslate2::StorageView& storage) {
      ctranslate2::StorageView cpu_storage = storage.to(ctranslate2::Device::CPU);
      return std::vector<T>(cpu_storage.data<T>(), cpu_storage.data<T>() + cpu_storage.size());
  }


  ctranslate2::Device _device = ctranslate2::Device::METAL;
  ctranslate2::Allocator* _allocator = nullptr;
};


// TODO: Add actual test cases here, e.g., TEST_F(MetalAttentionTest, BasicAttention)

TEST_F(MetalAttentionTest, BasicAttention) {
  // Define shapes and parameters
  const ctranslate2::dim_t batch_size = 1;
  const ctranslate2::dim_t num_heads = 1;
  const ctranslate2::dim_t q_len = 2;
  const ctranslate2::dim_t kv_len = 3;
  const ctranslate2::dim_t d_head = 4;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
  const bool causal = false;

  const ctranslate2::Shape q_shape = {batch_size, num_heads, q_len, d_head};
  const ctranslate2::Shape kv_shape = {batch_size, num_heads, kv_len, d_head};
  const ctranslate2::Shape out_shape = {batch_size, num_heads, q_len, d_head};

  // Define input data (CPU)
  const std::vector<float> q_vec = {1, 0, 0, 0,  0, 1, 0, 0}; // Shape: {1, 1, 2, 4}
  const std::vector<float> k_vec = {1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0}; // Shape: {1, 1, 3, 4}
  const std::vector<float> v_vec = {1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12}; // Shape: {1, 1, 3, 4}

  // Define expected output data (CPU)
  // Calculated manually/using reference implementation
  const std::vector<float> expected_out_vec = {
    4.28872f, 5.28872f, 6.28872f, 7.28872f, // Output for query 1
    5.0f,     6.0f,     7.0f,     8.0f      // Output for query 2
  };

  // Create Metal storage
  ctranslate2::StorageView q_metal = create_metal_storage(q_shape, q_vec);
  ctranslate2::StorageView k_metal = create_metal_storage(kv_shape, k_vec);
  ctranslate2::StorageView v_metal = create_metal_storage(kv_shape, v_vec);
  ctranslate2::StorageView out_metal({out_shape}, 0.f, ctranslate2::Device::METAL);

  // Call the attention primitive
  ctranslate2::primitives<ctranslate2::Device::METAL>::attention(
      out_metal,
      q_metal,
      k_metal,
      v_metal,
      nullptr, // mask - Not testing mask yet
      nullptr, // query_lengths - Assuming full length
      nullptr, // key_lengths - Assuming full length
      scale,
      num_heads,
      causal
  );

  // Get result back to CPU
  std::vector<float> actual_out_vec = get_vector<float>(out_metal);

  // Check result against expected values
  ASSERT_EQ(actual_out_vec.size(), expected_out_vec.size());
  for (size_t i = 0; i < actual_out_vec.size(); ++i) {
      ASSERT_NEAR(actual_out_vec[i], expected_out_vec[i], 1e-5f)
          << "Mismatch at index " << i;
  }
}



