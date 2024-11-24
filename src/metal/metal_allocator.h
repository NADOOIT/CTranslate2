#pragma once

#include "ctranslate2/allocator.h"
#include "metal_device.h"

namespace ctranslate2 {
  namespace metal {

    class MetalAllocator : public Allocator {
    public:
      explicit MetalAllocator(MetalDevice& device);
      ~MetalAllocator() override = default;

      void* allocate(std::size_t size, int device_index = -1) override;
      void free(void* data, int device_index = -1) override;
      void clear_cache() override;

      // Additional Metal-specific methods
      void* host_to_device(const void* host_data,
                          std::size_t size,
                          int device_index = -1);
      void device_to_host(const void* device_data,
                         void* host_data,
                         std::size_t size,
                         int device_index = -1);
      void device_to_device(const void* device_data,
                           void* other_device_data,
                           std::size_t size,
                           int device_index = -1,
                           int other_device_index = -1);

    private:
      MetalDevice& _device;
    };

  }  // namespace metal
}  // namespace ctranslate2
