#include "gemm_utils.h"
#include "gemm_bench_helpers.h"
#include <tuple>
#include <vector>
#include <fstream>
#include <iostream>
#include <Metal/Metal.h>

int main(int argc, char** argv) {
    std::string commit = "local";
    std::string operator_name = "gemm";
    std::string csv_file = "gemm_metal_bench.csv";
    std::vector<std::tuple<int,int,int>> sizes = { {32,32,32}, {128,128,128}, {512,512,512} };
    std::vector<int> batch_sizes = {1, 8, 32};
    int repeats = 10;
    bool exists = file_exists(csv_file);
    std::ofstream csv(csv_file, std::ios::app);
    if (!exists) {
        write_csv_header(csv);
    }
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        NSString* device_name_ns = [device name];
        std::string device_name = [device_name_ns UTF8String];
        run_metal_batched_benchmarks(sizes, batch_sizes, repeats, csv, commit, operator_name, device, device_name);
    }
    std::cout << "Metal GEMM benchmarks complete. Results in " << csv_file << std::endl;
    return 0;
}
