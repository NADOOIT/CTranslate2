#pragma once

#include <string>
#include <stdexcept>
#ifdef __OBJC__
#include <Metal/Metal.h>
#endif

namespace ctranslate2 {
  namespace metal {

    class MetalError : public std::runtime_error {
    public:
      explicit MetalError(const std::string& msg) : std::runtime_error(msg) {}
    };

    // Check if Metal is available
    bool has_metal();

    // Get number of Metal devices
    int get_metal_device_count();

\
    // Get current Metal device (Objective-C++ only)
#ifdef __OBJC__
    id<MTLDevice> get_metal_device();
#endif

    // Set current Metal device
    void set_metal_device(int index);

    // Initialize Metal device
    void init_metal();

    // Create Metal command queue (Objective-C++ only)
#ifdef __OBJC__
    id<MTLCommandQueue> create_command_queue();
#endif

    // Synchronize Metal device
    void synchronize_device();

    // Memory management
    void* metal_malloc(size_t size);
    void metal_free(void* ptr);
    void metal_memcpy(void* dst, const void* src, size_t size, bool to_device);

  }
}
