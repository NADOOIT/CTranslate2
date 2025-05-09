#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_allocator.h"
#include "ctranslate2/logging.h"

namespace ctranslate2 {
  namespace metal {

    MetalAllocator::MetalAllocator(MetalDevice& device)
      : _device(device) {
    }

    void* MetalAllocator::allocate(std::size_t size, int device_index) {
      return _device.allocate(size);
    }

    void MetalAllocator::free(void* data, int device_index) {
      _device.free(data);
    }

    void MetalAllocator::clear_cache() {
      // No caching implemented yet
    }

    void* MetalAllocator::host_to_device(const void* host_data,
                                        std::size_t size,
                                        int device_index) {
      void* device_buffer = allocate(size, device_index);
      id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)device_buffer;
      void* mapped_data = [buffer contents];
      std::memcpy(mapped_data, host_data, size);
      return device_buffer;
    }

    void MetalAllocator::device_to_host(const void* device_data,
                                       void* host_data,
                                       std::size_t size,
                                       int device_index) {
      id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)device_data;
      void* mapped_data = [buffer contents];
      std::memcpy(host_data, mapped_data, size);
    }

    void MetalAllocator::device_to_device(const void* device_data,
                                         void* other_device_data,
                                         std::size_t size,
                                         int device_index,
                                         int other_device_index) {
      id<MTLBuffer> src_buffer = (__bridge id<MTLBuffer>)device_data;
      id<MTLBuffer> dst_buffer = (__bridge id<MTLBuffer>)other_device_data;
      
      void* src_mapped = [src_buffer contents];
      void* dst_mapped = [dst_buffer contents];
      std::memcpy(dst_mapped, src_mapped, size);
    }

  }  // namespace metal
}  // namespace ctranslate2
