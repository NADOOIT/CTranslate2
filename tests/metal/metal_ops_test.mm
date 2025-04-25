#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            std::cout << "[MetalOpsTest] Metal-Framework ist grundsätzlich verfügbar." << std::endl;
            return 0;
        } else {
            std::cerr << "[MetalOpsTest] Metal-Framework NICHT verfügbar!" << std::endl;
            return 1;
        }
    }
}
