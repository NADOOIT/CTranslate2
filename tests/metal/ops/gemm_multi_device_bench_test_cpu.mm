#include "gemm_utils.h"
#include "gemm_cpu_accel.h"
#include "gemm_bench_helpers.h"
#include <tuple>
#include <vector>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    std::string commit = "local";
    std::string operator_name = "gemm";
    std::string csv_file = "gemm_cpu_bench.csv";
    std::vector<std::tuple<int,int,int>> sizes = { {32,32,32}, {128,128,128}, {512,512,512} };
    std::vector<int> batch_sizes = {1, 8, 32};
    int repeats = 10;
    bool exists = file_exists(csv_file);
    std::ofstream csv(csv_file, std::ios::app);
    if (!exists) {
        write_csv_header(csv);
    }
    // CPU-only: use a dummy device and name
    id<MTLDevice> dummy_device = nil;
    std::string device_name = "CPU";
    run_cpu_batched_benchmarks(sizes, batch_sizes, repeats, csv, commit, operator_name, dummy_device, device_name);
    std::cout << "CPU GEMM benchmarks complete. Results in " << csv_file << std::endl;
    return 0;
}
