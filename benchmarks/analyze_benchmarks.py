import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "benchmarks_ops.csv"
    df = pd.read_csv(csv_file)
    # Only keep relevant columns
    keep_cols = ["variant", "size", "cpu_ms", "metal_ms", "speedup"]
    df = df[keep_cols]
    # Compute speedup for each row (if both cpu_ms and metal_ms are available)
    df = df.dropna(subset=["cpu_ms", "metal_ms"])
    df["speedup_vs_cpu"] = df["cpu_ms"] / df["metal_ms"]
    # Pivot for plotting: show speedup by variant and size
    pivot = df.pivot_table(index="variant", columns="size", values="speedup_vs_cpu", aggfunc="max")
    pivot.plot(kind="bar", figsize=(12,6))
    plt.title("GEMM Speedup: Metal vs. CPU")
    plt.ylabel("Speedup (X)")
    plt.xlabel("Variant")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("gemm_speedup_vs_cpu.png")
    plt.show()

if __name__ == "__main__":
    main()
