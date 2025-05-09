#include "metal_utils.h"
#include <stdexcept>
#include <iostream>

namespace ctranslate2 {
  namespace metal {

    MetalUtils::MetalUtils()
      : _device(nil)
      , _command_queue(nil) {
      
      @autoreleasepool {
        // Try to list all available devices first
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (!devices || [devices count] == 0) {
          std::cerr << "MTLCopyAllDevices returned no devices" << std::endl;
          throw std::runtime_error("No Metal devices available");
        }

        std::cerr << "Found " << [devices count] << " Metal device(s):" << std::endl;
        for (NSUInteger i = 0; i < [devices count]; i++) {
          id<MTLDevice> device = devices[i];
          std::cerr << "  Device " << i << ": " << [device.name UTF8String] 
                    << " (Headless: " << (device.headless ? "Yes" : "No")
                    << ", Low Power: " << (device.lowPower ? "Yes" : "No") << ")" << std::endl;
        }

        // Select the first device that is not low-power
        _device = nil;
        for (id<MTLDevice> device in devices) {
          if (!device.lowPower) {
            _device = device;
            break;
          }
        }

        // If no high-performance device found, just use the first one
        if (!_device && [devices count] > 0) {
          _device = devices[0];
        }

        if (!_device) {
          std::cerr << "Failed to select a Metal device" << std::endl;
          throw std::runtime_error("Failed to select Metal device");
        }

        // Create command queue with error handling
        _command_queue = [_device newCommandQueue];
        if (!_command_queue) {
          std::cerr << "Failed to create Metal command queue" << std::endl;
          throw std::runtime_error("Failed to create Metal command queue");
        }

        std::cerr << "\nSelected Metal device:" << std::endl;
        std::cerr << "  Name: " << [_device.name UTF8String] << std::endl;
        std::cerr << "  Registry ID: 0x" << std::hex << _device.registryID << std::dec << std::endl;
        std::cerr << "  Location: " << (_device.removable ? "External" : "Internal") << std::endl;
        std::cerr << "  Low-power: " << (_device.lowPower ? "Yes" : "No") << std::endl;
        std::cerr << "  Headless: " << (_device.headless ? "Yes" : "No") << std::endl;

        // Check feature sets
        if (@available(macOS 10.11, *)) {
          std::cerr << "\nFeature Set Support:" << std::endl;
          std::cerr << "  macOS GPU Family 1 v1: " 
                    << ([_device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily1_v1] ? "Yes" : "No") 
                    << std::endl;
          
          if (@available(macOS 10.12, *)) {
            std::cerr << "  macOS GPU Family 1 v2: " 
                      << ([_device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily1_v2] ? "Yes" : "No") 
                      << std::endl;
          }
          
          if (@available(macOS 10.13, *)) {
            std::cerr << "  macOS GPU Family 1 v3: " 
                      << ([_device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily1_v3] ? "Yes" : "No") 
                      << std::endl;
          }
          
          if (@available(macOS 10.14, *)) {
            std::cerr << "  macOS GPU Family 1 v4: " 
                      << ([_device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily1_v4] ? "Yes" : "No") 
                      << std::endl;
          }
        }

        std::cerr << "\nSuccessfully initialized Metal device and command queue" << std::endl;
      }
    }

    MetalUtils::~MetalUtils() {
      // ARC will handle cleanup
      _command_queue = nil;
      _device = nil;
    }

  }  // namespace metal
}  // namespace ctranslate2
