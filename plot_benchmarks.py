import pandas as pd
import matplotlib.pyplot as plt
import os

# Set paths to CSV files (update if needed)
root = os.path.dirname(os.path.abspath(__file__))
bench_ops_path = os.path.join(root, 'tests/metal/ops/build/benchmarks_ops.csv')
gemm_cpu_path = os.path.join(root, 'tests/metal/ops/build/gemm_cpu_bench.csv')
gemm_metal_path = os.path.join(root, 'tests/metal/ops/build/gemm_metal_bench.csv')

def plot_ops_benchmarks():
    df = pd.read_csv(bench_ops_path)
    # Plot GEMM speedup
    df_gemm = df[df['operator'] == 'GEMM']
    plt.figure(figsize=(8, 5))
    plt.title('GEMM: Metal vs CPU Speedup')
    plt.bar(df_gemm['size'], df_gemm['speedup'], color='royalblue')
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup (CPU ms / Metal ms)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    # Plot ReLU speedup
    df_relu = df[df['operator'] == 'ReLU']
    plt.figure(figsize=(8, 5))
    plt.title('ReLU: Metal vs CPU Speedup')
    plt.bar(df_relu['size'], df_relu['speedup'], color='orange')
    plt.xlabel('Input Size')
    plt.ylabel('Speedup (CPU ms / Metal ms)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

def plot_gemm_cpu():
    df = pd.read_csv(gemm_cpu_path)
    # Only plot a subset for clarity
    sizes = df['size'].unique()
    for size in sizes:
        df_size = df[df['size'] == size]
        plt.plot(df_size['batch'], df_size['avg_ms'], marker='o', label=f'Size {size}')
    plt.title('CPU GEMM: Batch Size vs Time')
    plt.xlabel('Batch Size')
    plt.ylabel('Avg Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    plot_ops_benchmarks()
    plot_gemm_cpu()
    # You can add more plots for gemm_metal_bench.csv if data is available

if __name__ == '__main__':
    main()
