# GEMM Benchmark Suite: Apple Silicon Maximum Performance

## Overview
This benchmark suite measures the performance of GEMM (General Matrix Multiply) operations across all available compute backends on Apple Silicon (CPU, Accelerate/BLAS, Metal, Metal Tiled, Metal Performance Shaders, and optionally ANE and FP16). It provides a reproducible framework for comparing performance, logging results, and visualizing speedups.

## Benchmark Modes
- `cpu_naive`: Single-threaded C++ GEMM
- `cpu_accelerate`: Apple Accelerate (BLAS) single-threaded
- `cpu_gcd`: Multi-threaded GEMM using Grand Central Dispatch
- `cpu_accgcd`: Accelerate+GCD (multi-threaded BLAS)
- `metal`: Basic Metal kernel
- `metal_batched`: Multiple Metal command buffers in flight
- `metal_tiled`: Tiled/shared-memory Metal kernel (highly optimized)
- `mps`: Metal Performance Shaders GEMM
- `metal_fp16` (optional): Metal kernel in FP16
- `mps_fp16` (optional): MPS GEMM in FP16
- `ane`: Apple Neural Engine (if available)
- `hybrid`: Batch split across CPU, GPU, and ANE in parallel (see below)

## How to Build and Run

1. **Build**
   ```sh
   cd /Users/christophbackhaus/Documents/GitHub/CTranslate2/build/tests/metal/ops
   make -j$(sysctl -n hw.logicalcpu)
   ```

2. **Run**
   ```sh
   ./gemm_multi_device_bench_test
   ```
   Results are logged to `benchmarks_ops.csv`.

3. **Analyze**
   Use the provided Python script to generate plots and compare speedups.

## CSV Output Format
```
timestamp,commit,operator,mode,device,size,batch,avg_ms,status
```
For hybrid mode, the CSV includes per-device timing:
```
timestamp,commit,operator,mode,device,size,batch,cpu_ms,gpu_ms,ane_ms,total_ms,status
```

## Plotting and Analysis
Run the provided Python script:
```sh
python3 analyze_benchmarks.py
```

## Device Selection
The harness automatically detects and logs results for all available Metal devices (GPU, ANE, etc.).

## Hybrid Batching Mode (CPU + GPU + ANE)

The `hybrid` mode splits large batches across all available devices (CPU, GPU, ANE) to maximize throughput. Each device processes a portion of the batch in parallel. Timings for each device and the total are logged in the CSV for full transparency.

- **How it works:**
  - The batch is divided among CPU, GPU, and ANE based on availability and (optionally) device throughput.
  - GEMM runs in parallel on each device using separate threads.
  - The CSV logs per-device times (`cpu_ms`, `gpu_ms`, `ane_ms`) and the overall `total_ms`.
- **Interpreting Results:**
  - Compare `total_ms` to single-device runs to see the benefit of hybrid parallelism.
  - Use the per-device columns to identify bottlenecks and optimize batch splitting.

## Extending
- Add new kernels or modes by editing the harness and kernels.
- Add new matrix sizes or batch sizes as needed.

## Contact
For questions or contributions, open an issue or pull request on GitHub.
