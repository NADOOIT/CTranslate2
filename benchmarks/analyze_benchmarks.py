import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_absolute_latency(df):
    import matplotlib.pyplot as plt
    pivot = df.pivot_table(index="variant", columns="size", values="metal_ms", aggfunc="min")
    pivot.T.plot(marker='o', figsize=(12,6))
    plt.title("Absolute Metal Latency (ms) by Variant and Matrix Size")
    plt.ylabel("Latency (ms)")
    plt.xlabel("Matrix Size")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("gemm_absolute_latency.png")
    plt.show()

def plot_speedup_heatmap(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    pivot = df.pivot_table(index="variant", columns="size", values="speedup_vs_cpu", aggfunc="max")
    plt.figure(figsize=(10,7))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Speedup Heatmap: Metal vs. CPU")
    plt.ylabel("Variant")
    plt.xlabel("Matrix Size")
    plt.tight_layout()
    plt.savefig("gemm_speedup_heatmap.png")
    plt.show()

def plot_batch_scaling(df):
    import matplotlib.pyplot as plt
    if "batch" in df.columns:
        for variant in df["variant"].unique():
            df_v = df[df["variant"]==variant]
            if not df_v.empty:
                plt.plot(df_v["batch"], df_v["metal_ms"], marker='o', label=variant)
        plt.title("Batch Size Scaling (Metal)")
        plt.xlabel("Batch Size")
        plt.ylabel("Metal Latency (ms)")
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig("gemm_batch_scaling.png")
        plt.show()

def plot_hybrid_breakdown(df):
    import matplotlib.pyplot as plt
    # Only plot if hybrid results exist
    if "hybrid" in df["variant"].str.lower().values:
        df_hybrid = df[df["variant"].str.lower().str.contains("hybrid")]
        if not df_hybrid.empty:
            for idx, row in df_hybrid.iterrows():
                labels = []
                values = []
                for col in ["cpu_ms", "metal_ms", "ane_ms"]:
                    if col in row and not pd.isnull(row[col]):
                        labels.append(col)
                        values.append(row[col])
                plt.figure()
                plt.bar(labels, values)
                plt.title(f"Hybrid Breakdown (size={row['size']})")
                plt.ylabel("Latency (ms)")
                plt.tight_layout()
                plt.savefig(f"gemm_hybrid_breakdown_{row['size']}.png")
                plt.show()

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "benchmarks_ops.csv"
    df = pd.read_csv(csv_file)
    # Only keep relevant columns that might exist
    keep_cols = [col for col in ["variant", "size", "cpu_ms", "metal_ms", "ane_ms", "batch", "speedup"] if col in df.columns]
    df = df[keep_cols]
    # Compute speedup for each row (if both cpu_ms and metal_ms are available)
    if "cpu_ms" in df.columns and "metal_ms" in df.columns:
        df = df.dropna(subset=["cpu_ms", "metal_ms"])
        df["speedup_vs_cpu"] = df["cpu_ms"] / df["metal_ms"]
    # Pivot for plotting: show speedup by variant and size
    if "speedup_vs_cpu" in df.columns:
        pivot = df.pivot_table(index="variant", columns="size", values="speedup_vs_cpu", aggfunc="max")
        pivot.plot(kind="bar", figsize=(12,6))
        plt.title("GEMM Speedup: Metal vs. CPU")
        plt.ylabel("Speedup (X)")
        plt.xlabel("Variant")
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig("gemm_speedup_vs_cpu.png")
        plt.show()
        plot_speedup_heatmap(df)
    plot_absolute_latency(df)
    plot_batch_scaling(df)
    plot_hybrid_breakdown(df)

if __name__ == "__main__":
    main()
