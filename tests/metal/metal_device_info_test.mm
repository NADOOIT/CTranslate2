#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <fstream>
#include <ctime>

int main() {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices;
        if (@available(macOS 10.13, *)) {
            devices = MTLCopyAllDevices();
        } else {
            devices = @[MTLCreateSystemDefaultDevice()];
        }
        std::ofstream csv("benchmarks_ops.csv", std::ios::app);
        // Schreibe Header, falls Datei neu
        std::ifstream check("benchmarks_ops.csv");
        bool file_exists = check.good();
        check.close();
        if (!file_exists) {
            csv << "timestamp,commit,operator,variant,size,cpu_ms,metal_ms,speedup\n";
        }
        std::time_t t = std::time(nullptr);
        char timebuf[32];
        std::strftime(timebuf, sizeof(timebuf), "%FT%T%z", std::localtime(&t));
        for (id<MTLDevice> device in devices) {
            NSString* name = [device name];
            bool isLowPower = [device respondsToSelector:@selector(isLowPower)] ? [device isLowPower] : false;
            bool isRemovable = [device respondsToSelector:@selector(isRemovable)] ? [device isRemovable] : false;
            bool headless = [device respondsToSelector:@selector(isHeadless)] ? [device isHeadless] : false;
            NSUInteger maxThreads = [device maxThreadsPerThreadgroup].width * [device maxThreadsPerThreadgroup].height * [device maxThreadsPerThreadgroup].depth;
            bool isNeuralEngine = false;
            // Apple Neural Engine Detection (Workaround):
            if ([name containsString:@"ANE"] || [name containsString:@"Neural"]) {
                isNeuralEngine = true;
            }
            std::cout << "[DeviceInfo] Name: " << [name UTF8String]
                      << ", LowPower: " << isLowPower
                      << ", Removable: " << isRemovable
                      << ", Headless: " << headless
                      << ", MaxThreads: " << maxThreads
                      << ", NeuralEngine: " << isNeuralEngine << std::endl;
            // Schreibe ins Benchmark-CSV
            csv << timebuf << ",f35fc96e,DEVICE_INFO," << (isNeuralEngine ? "NeuralEngine" : "MetalGPU") << "," << [name UTF8String] << ",0,0,0\n";
        }
        return 0;
    }
}
