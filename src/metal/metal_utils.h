#pragma once

#include <Metal/Metal.h>

namespace ctranslate2 {
  namespace metal {

    class MetalUtils {
    public:
      static MetalUtils& getInstance() {
        static MetalUtils instance;
        return instance;
      }

      id<MTLDevice> getDevice() const { return _device; }
      id<MTLCommandQueue> getCommandQueue() const { return _command_queue; }

      // Delete copy constructor and assignment operator
      MetalUtils(const MetalUtils&) = delete;
      MetalUtils& operator=(const MetalUtils&) = delete;

    private:
      MetalUtils();  // Private constructor for singleton
      ~MetalUtils();

      id<MTLDevice> _device;
      id<MTLCommandQueue> _command_queue;
    };

  }  // namespace metal
}  // namespace ctranslate2
