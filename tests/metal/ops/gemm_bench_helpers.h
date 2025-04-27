#pragma once
#include <vector>
#include <string>
#include <tuple>
#include <fstream>
#include <Metal/Metal.h>

// Batched GEMM benchmark for CPU, Accelerate, GCD, Accelerate+GCD
void run_cpu_batched_benchmarks(const std::vector<std::tuple<int,int,int>>& sizes,
                                const std::vector<int>& batch_sizes,
                                int repeats,
                                std::ofstream& csv,
                                const std::string& commit,
                                const std::string& operator_name,
                                id<MTLDevice> device,
                                const std::string& device_name);

// Batched GEMM benchmark for Metal
void run_metal_batched_benchmarks(const std::vector<std::tuple<int,int,int>>& sizes,
                                  const std::vector<int>& batch_sizes,
                                  int repeats,
                                  std::ofstream& csv,
                                  const std::string& commit,
                                  const std::string& operator_name,
                                  id<MTLDevice> device,
                                  const std::string& device_name);
