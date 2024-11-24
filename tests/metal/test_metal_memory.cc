#include <gtest/gtest.h>
#include <vector>
#include <thread>
#include <chrono>

#include "ctranslate2/devices.h"
#include "ctranslate2/allocator.h"
#include "metal/metal_device.h"
#include "metal/metal_allocator.h"
#include "metal/metal_kernels.h"

using namespace ctranslate2;

class MetalMemoryTest : public ::testing::Test {
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

// Test repeated allocation and deallocation
TEST_F(MetalMemoryTest, RepeatedAllocation) {
  const size_t size = 1024 * 1024;  // 1MB
  const int iterations = 100;

  for (int i = 0; i < iterations; ++i) {
    void* ptr = allocator->allocate(size);
    ASSERT_NE(ptr, nullptr);
    allocator->free(ptr);
  }
}

// Test multiple allocations without immediate deallocation
TEST_F(MetalMemoryTest, MultipleAllocations) {
  const size_t size = 1024 * 1024;  // 1MB
  const int num_allocations = 10;
  std::vector<void*> ptrs;

  for (int i = 0; i < num_allocations; ++i) {
    void* ptr = allocator->allocate(size);
    ASSERT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  for (void* ptr : ptrs) {
    allocator->free(ptr);
  }
}

// Test allocation stress with varying sizes
TEST_F(MetalMemoryTest, StressTest) {
  const int iterations = 50;
  std::vector<std::pair<void*, size_t>> allocations;

  for (int i = 0; i < iterations; ++i) {
    size_t size = (1 << (i % 20)) * 1024;  // Vary size from 1KB to 1GB
    void* ptr = allocator->allocate(size);
    ASSERT_NE(ptr, nullptr);
    allocations.emplace_back(ptr, size);

    // Randomly free some allocations
    if (i % 3 == 0 && !allocations.empty()) {
      size_t index = i % allocations.size();
      allocator->free(allocations[index].first);
      allocations.erase(allocations.begin() + index);
    }
  }

  // Free remaining allocations
  for (const auto& alloc : allocations) {
    allocator->free(alloc.first);
  }
}

// Test concurrent allocations
TEST_F(MetalMemoryTest, ConcurrentAllocations) {
  const size_t size = 1024 * 1024;  // 1MB
  const int num_threads = 4;
  const int iterations_per_thread = 25;

  auto thread_func = [this, size, iterations_per_thread]() {
    std::vector<void*> ptrs;
    for (int i = 0; i < iterations_per_thread; ++i) {
      void* ptr = allocator->allocate(size);
      ASSERT_NE(ptr, nullptr);
      ptrs.push_back(ptr);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    for (void* ptr : ptrs) {
      allocator->free(ptr);
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(thread_func);
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

// Test memory transfer operations
TEST_F(MetalMemoryTest, MemoryTransfer) {
  const size_t size = 1024;
  std::vector<float> host_data(size, 1.0f);

  // Test host to device transfer
  void* device_ptr = toDevice(host_data);
  ASSERT_NE(device_ptr, nullptr);

  // Test device to host transfer
  auto result = toHost<float>(device_ptr, size);
  ASSERT_EQ(result.size(), size);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(result[i], host_data[i]);
  }

  allocator->free(device_ptr);
}

// Test device-to-device memory operations
TEST_F(MetalMemoryTest, DeviceToDevice) {
  const size_t size = 1024;
  std::vector<float> host_data(size, 2.0f);

  void* src_ptr = toDevice(host_data);
  void* dst_ptr = allocator->allocate(size * sizeof(float));

  allocator->device_to_device(src_ptr, dst_ptr, size * sizeof(float));

  auto result = toHost<float>(dst_ptr, size);
  ASSERT_EQ(result.size(), size);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(result[i], host_data[i]);
  }

  allocator->free(src_ptr);
  allocator->free(dst_ptr);
}

// Test allocation failure handling
TEST_F(MetalMemoryTest, AllocationFailure) {
  const size_t huge_size = size_t(1) << 40;  // 1TB
  EXPECT_THROW(allocator->allocate(huge_size), std::runtime_error);
}

// Test memory alignment
TEST_F(MetalMemoryTest, MemoryAlignment) {
  const size_t sizes[] = {1, 3, 7, 13, 32, 64, 128, 256};
  
  for (size_t size : sizes) {
    void* ptr = allocator->allocate(size);
    ASSERT_NE(ptr, nullptr);
    
    // Metal buffers should be aligned to at least 16 bytes
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % 16, 0) << "Size " << size << " not aligned to 16 bytes";
    
    allocator->free(ptr);
  }
}

// Test the allocator's behavior with zero-sized allocations
TEST_F(MetalMemoryTest, ZeroSizeAllocation) {
  void* ptr = allocator->allocate(0);
  // Implementation-defined: either return nullptr or a valid pointer
  if (ptr != nullptr) {
    allocator->free(ptr);
  }
}

// Test memory operations with streams
TEST_F(MetalMemoryTest, StreamOperations) {
  const size_t size = 1024;
  std::vector<float> host_data(size, 3.0f);

  void* stream = device->allocate_stream();
  ASSERT_NE(stream, nullptr);

  void* device_ptr = toDevice(host_data);
  ASSERT_NE(device_ptr, nullptr);

  // Perform some operations with the stream
  metal::add(*device,
            static_cast<const float*>(device_ptr),
            static_cast<const float*>(device_ptr),
            static_cast<float*>(device_ptr),
            size,
            stream);

  device->synchronize_stream(stream);

  auto result = toHost<float>(device_ptr, size);
  ASSERT_EQ(result.size(), size);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(result[i], host_data[i] * 2.0f);
  }

  allocator->free(device_ptr);
  device->free_stream(stream);
}
