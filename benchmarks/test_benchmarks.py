import os
import subprocess
import pandas as pd
import pytest

def test_benchmark_csv_exists():
    # Try default locations
    candidates = [
        'benchmarks_ops.csv',
        '../build/tests/metal/ops/benchmarks_ops.csv',
        '../../build/tests/metal/ops/benchmarks_ops.csv',
    ]
    found = False
    for path in candidates:
        if os.path.exists(path):
            found = path
            break
    assert found, 'benchmarks_ops.csv not found in any expected location.'
    return found

def test_benchmark_csv_content():
    csv_path = test_benchmark_csv_exists()
    df = pd.read_csv(csv_path)
    # Check columns
    # Match actual columns
    assert set(['timestamp','commit','operator','variant','size','cpu_ms','metal_ms','speedup']).issubset(df.columns)
    # There should be at least one Metal result
    assert (df['variant'].str.lower().str.contains('metal') | df['variant'].str.lower().str.contains('gpu')).any(), 'No Metal or GPU results found.'
    # Speedup check: for large matrices, Metal should be faster than CPU
    df_large = df[df['size'].str.contains('512') & (df['variant'] == 'metal')]
    df_cpu = df[df['size'].str.contains('512') & (df['variant'] == 'cpu_naive')]
    if not df_large.empty and not df_cpu.empty:
        metal_ms = df_large.iloc[0]['metal_ms']
        cpu_ms = df_cpu.iloc[0]['cpu_ms']
        assert metal_ms < cpu_ms, f'Metal not faster than CPU for large matrix: {metal_ms} vs {cpu_ms}'

if __name__ == '__main__':
    pytest.main([__file__])
