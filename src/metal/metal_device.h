#pragma once

#include <memory>
#include <string>

#include "ctranslate2/devices.h"
#include "metal_utils.h"

namespace ctranslate2 {
  namespace metal {

    class MetalDevice {
    public:
      MetalDevice(int index = 0);
      ~MetalDevice();

      Device type() const {
        return Device::METAL;
      }

      int index() const {
        return _device_index;
      }

      bool support_stream() const {
        return true;
      }

      std::string name() const;

      void synchronize();
      void synchronize_stream(void* stream);

      void* allocate(std::size_t size) const;
      void free(void* data) const;
      void* allocate_stream() const;
      void free_stream(void* stream) const;

      // Get Metal utilities
      const MetalUtils& getMetalUtils() const {
        return _utils;
      }

    private:
      int _device_index;
      void* _command_queue;  // MTLCommandQueue*
      MetalUtils& _utils;  // Reference to singleton MetalUtils
    };

  }  // namespace metal
}  // namespace ctranslate2
