#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "[MetalBufferTest] Kein Metal-Device gefunden!" << std::endl;
            return 1;
        }
        std::cout << "[MetalBufferTest] Metal-Device Name: " << [[device name] UTF8String] << std::endl;
        NSUInteger maxBufferLength = [device maxBufferLength];
        std::cout << "[MetalBufferTest] Max Buffer Length: " << maxBufferLength << std::endl;

        NSUInteger bufferSize = 1024;
        id<MTLBuffer> buffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        if (!buffer) {
            std::cerr << "[MetalBufferTest] Buffer konnte nicht angelegt werden!" << std::endl;
            return 2;
        }
        std::cout << "[MetalBufferTest] Buffer erfolgreich angelegt." << std::endl;
        return 0;
    }
}
