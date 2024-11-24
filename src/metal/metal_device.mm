#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_device.h"
#include "ctranslate2/logging.h"
#include <iostream>

namespace ctranslate2 {
  namespace metal {

    MetalDevice::MetalDevice(int index)
      : _device_index(index)
      , _utils(MetalUtils::getInstance())
      , _command_queue(nullptr) {
      
      @autoreleasepool {
        // Get device and command queue from MetalUtils
        id<MTLDevice> device = _utils.getDevice();
        id<MTLCommandQueue> queue = _utils.getCommandQueue();

        if (!device || !queue) {
          std::cerr << "Failed to get Metal device or command queue from MetalUtils" << std::endl;
          throw std::runtime_error("Failed to create Metal device");
        }

        // Store command queue reference
        _command_queue = (__bridge_retained void*)queue;
      }
    }

    MetalDevice::~MetalDevice() {
      if (_command_queue) {
        CFRelease(_command_queue);
        _command_queue = nullptr;
      }
    }

    std::string MetalDevice::name() const {
      @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLDevice> device = queue.device;
        return [device.name UTF8String];
      }
    }

    void MetalDevice::synchronize() {
      @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
      }
    }

    void MetalDevice::synchronize_stream(void* stream) {
      if (stream) {
        @autoreleasepool {
          id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)stream;
          [commandBuffer waitUntilCompleted];
        }
      }
    }

    void* MetalDevice::allocate(std::size_t size) const {
      if (size == 0) {
        return nullptr;
      }

      @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLDevice> device = queue.device;
        
        // Use shared storage mode for better compatibility
        MTLResourceOptions options = MTLResourceStorageModeShared;
        
        // Create buffer
        id<MTLBuffer> buffer = [device newBufferWithLength:size options:options];
        
        if (!buffer) {
          std::string errorMsg = "Failed to allocate Metal buffer of size " + std::to_string(size);
          std::cerr << errorMsg << std::endl;
          throw std::runtime_error(errorMsg);
        }
        
        return (__bridge_retained void*)buffer;
      }
    }

    void MetalDevice::free(void* data) const {
      if (data) {
        CFRelease(data);
      }
    }

    void* MetalDevice::allocate_stream() const {
      @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        return commandBuffer ? (__bridge_retained void*)commandBuffer : nullptr;
      }
    }

    void MetalDevice::free_stream(void* stream) const {
      if (stream) {
        CFRelease(stream);
      }
    }

  }  // namespace metal
}  // namespace ctranslate2
