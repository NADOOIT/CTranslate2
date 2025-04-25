#include <iostream>
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

int main() {
#if defined(__APPLE__) && defined(__OBJC__)
    std::cout << "[MetalOpsTest] Metal-Framework ist grundsätzlich verfügbar." << std::endl;
    // Hier könnten echte Metal-API-Aufrufe stehen (z.B. MTLCreateSystemDefaultDevice)
    return 0;
#else
    std::cerr << "[MetalOpsTest] Metal-Framework NICHT verfügbar!" << std::endl;
    return 1;
#endif
}
