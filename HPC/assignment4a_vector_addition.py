"""
Assignment 4A: Vector Addition using GPU simulation with NumPy
Note: This simulates CUDA-style vector addition in Python using NumPy
For actual CUDA implementation, use the .cu files as reference
"""

import time
import numpy as np
from typing import Tuple


def vector_add_cpu_sequential(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sequential vector addition on CPU"""
    n = len(a)
    c = np.zeros(n, dtype=np.float32)

    for i in range(n):
        c[i] = a[i] + b[i]

    return c


def vector_add_cpu_parallel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized (parallel-like) addition using NumPy"""
    return a + b


def simulate_gpu_vector_add(a: np.ndarray, b: np.ndarray, threads_per_block: int = 256) -> np.ndarray:
    """
    Simulates GPU vector addition with thread blocks
    This mimics CUDA's parallel execution model
    """
    n = len(a)
    c = np.zeros(n, dtype=np.float32)

    # Calculate grid dimensions (like CUDA)
    num_blocks = (n + threads_per_block - 1) // threads_per_block

    # Simulate parallel execution by blocks
    for block_idx in range(num_blocks):
        # Each block processes threads_per_block elements
        start_idx = block_idx * threads_per_block
        end_idx = min(start_idx + threads_per_block, n)

        # Simulate parallel threads in this block
        c[start_idx:end_idx] = a[start_idx:end_idx] + b[start_idx:end_idx]

    return c


def benchmark_vector_addition(size: int):
    """Benchmark different vector addition implementations"""
    print(f"\n{'=' * 70}")
    print(f"Vector Addition Benchmark - Size: {size:,}")
    print(f"{'=' * 70}")

    # Initialize vectors
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)

    print(f"\nVector A first 5 elements: {a[:5]}")
    print(f"Vector B first 5 elements: {b[:5]}")
    print(f"Memory per vector: {a.nbytes / (1024*1024):.2f} MB")

    # Sequential CPU
    print(f"\n{'-' * 70}")
    print("Sequential CPU Vector Addition")
    print(f"{'-' * 70}")
    start = time.time()
    c_seq = vector_add_cpu_sequential(a, b)
    time_seq = time.time() - start
    print(f"Time: {time_seq:.6f} seconds")
    print(f"Result first 5 elements: {c_seq[:5]}")

    # Parallel CPU (NumPy vectorized)
    print(f"\n{'-' * 70}")
    print("Parallel CPU Vector Addition (NumPy)")
    print(f"{'-' * 70}")
    start = time.time()
    c_par = vector_add_cpu_parallel(a, b)
    time_par = time.time() - start
    print(f"Time: {time_par:.6f} seconds")
    print(f"Result first 5 elements: {c_par[:5]}")

    # GPU Simulation
    print(f"\n{'-' * 70}")
    print("GPU-Style Vector Addition (Simulated)")
    print(f"{'-' * 70}")
    threads_per_block = 256
    start = time.time()
    c_gpu = simulate_gpu_vector_add(a, b, threads_per_block)
    time_gpu = time.time() - start
    print(f"Threads per block: {threads_per_block}")
    print(f"Number of blocks: {(size + threads_per_block - 1) // threads_per_block}")
    print(f"Time: {time_gpu:.6f} seconds")
    print(f"Result first 5 elements: {c_gpu[:5]}")

    # Verification
    print(f"\n{'-' * 70}")
    print("Verification")
    print(f"{'-' * 70}")
    error_par = np.max(np.abs(c_seq - c_par))
    error_gpu = np.max(np.abs(c_seq - c_gpu))
    print(f"Max error (Parallel vs Sequential): {error_par:.10f}")
    print(f"Max error (GPU vs Sequential): {error_gpu:.10f}")
    print(f"All results correct: {error_par < 1e-5 and error_gpu < 1e-5}")

    # Performance summary
    print(f"\n{'-' * 70}")
    print("Performance Summary")
    print(f"{'-' * 70}")
    speedup_par = time_seq / time_par if time_par > 0 else 0
    speedup_gpu = time_seq / time_gpu if time_gpu > 0 else 0

    print(f"{'Method':<30} {'Time (s)':<15} {'Speedup':<10}")
    print("-" * 70)
    print(f"{'Sequential CPU':<30} {time_seq:<15.6f} {1.0:<10.2f}x")
    print(f"{'Parallel CPU (NumPy)':<30} {time_par:<15.6f} {speedup_par:<10.2f}x")
    print(f"{'GPU-style (Simulated)':<30} {time_gpu:<15.6f} {speedup_gpu:<10.2f}x")


def explain_cuda_concept():
    """Explain CUDA programming concepts"""
    print("\n" + "=" * 70)
    print("CUDA PROGRAMMING CONCEPTS")
    print("=" * 70)
    print("""
Vector addition is a perfect example for GPU parallelization because:
1. Each element operation is independent
2. Same operation applied to all elements (SIMD - Single Instruction Multiple Data)
3. No data dependencies between threads

CUDA Execution Model:
- Grid: Collection of blocks
- Block: Collection of threads (e.g., 256 threads per block)
- Thread: Individual computation unit

Example for vector of size 1000 with 256 threads/block:
- Number of blocks needed = ceil(1000/256) = 4 blocks
- Block 0: processes elements 0-255
- Block 1: processes elements 256-511
- Block 2: processes elements 512-767
- Block 3: processes elements 768-999

In actual CUDA C:
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

Each thread calculates its global index and processes one element.
    """)
    print("=" * 70)


def main():
    print("=" * 70)
    print("Assignment 4A: Large Vector Addition")
    print("GPU Simulation using Python/NumPy")
    print("=" * 70)

    explain_cuda_concept()

    # Test with different sizes
    sizes = [1_000_000, 10_000_000, 50_000_000]

    for size in sizes:
        benchmark_vector_addition(size)
        print()

    print("\n" + "=" * 70)
    print("NOTE: For actual GPU acceleration, use:")
    print("  - CUDA C (see .cu reference files)")
    print("  - CuPy (CUDA-accelerated NumPy)")
    print("  - Numba with @cuda.jit decorator")
    print("=" * 70)


if __name__ == "__main__":
    main()
