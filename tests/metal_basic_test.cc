#include <iostream>
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

int main() {
#if defined(__APPLE__) && defined(__OBJC__)
    std::cout << "[MetalTest] Metal support: Objective-C++ detected." << std::endl;
    // Hier könnte ein echter Metal-Aufruf stehen, z.B. Initialisierung eines Metal-Devices
    return 0;
#else
    std::cerr << "[MetalTest] Metal support NOT available!" << std::endl;
    return 1;
#endif
}
