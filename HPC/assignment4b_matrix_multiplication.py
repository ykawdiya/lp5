"""
Assignment 4B: Matrix Multiplication using GPU simulation with NumPy
Note: This simulates CUDA-style matrix multiplication in Python
For actual CUDA implementation, use the .cu files as reference
"""

import time
import numpy as np
from typing import Tuple


def matrix_multiply_cpu_sequential(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Sequential matrix multiplication on CPU (naive implementation)"""
    m, k = A.shape
    k2, n = B.shape

    assert k == k2, "Matrix dimensions must match for multiplication"

    C = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            sum_val = 0.0
            for p in range(k):
                sum_val += A[i, p] * B[p, j]
            C[i, j] = sum_val

    return C


def matrix_multiply_cpu_optimized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Optimized matrix multiplication using NumPy"""
    return np.matmul(A, B)


def simulate_gpu_matrix_multiply(A: np.ndarray, B: np.ndarray,
                                 block_size: int = 16) -> np.ndarray:
    """
    Simulates GPU matrix multiplication with 2D thread blocks
    This mimics CUDA's tiled matrix multiplication
    """
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, "Matrix dimensions must match"

    C = np.zeros((m, n), dtype=np.float32)

    # Calculate grid dimensions
    grid_rows = (m + block_size - 1) // block_size
    grid_cols = (n + block_size - 1) // block_size

    # Simulate execution by blocks
    for block_row in range(grid_rows):
        for block_col in range(grid_cols):
            # Each block computes a tile of C
            row_start = block_row * block_size
            row_end = min(row_start + block_size, m)
            col_start = block_col * block_size
            col_end = min(col_start + block_size, n)

            # Compute this tile
            for i in range(row_start, row_end):
                for j in range(col_start, col_end):
                    sum_val = 0.0
                    for p in range(k):
                        sum_val += A[i, p] * B[p, j]
                    C[i, j] = sum_val

    return C


def simulate_gpu_matrix_multiply_tiled(A: np.ndarray, B: np.ndarray,
                                       tile_size: int = 16) -> np.ndarray:
    """
    Simulates GPU tiled matrix multiplication
    This demonstrates shared memory optimization concept from CUDA
    """
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, "Matrix dimensions must match"

    C = np.zeros((m, n), dtype=np.float32)

    # Process tiles
    for i in range(0, m, tile_size):
        for j in range(0, n, tile_size):
            # Accumulate result for this tile of C
            for p in range(0, k, tile_size):
                # Extract tiles (simulating shared memory load)
                i_end = min(i + tile_size, m)
                j_end = min(j + tile_size, n)
                p_end = min(p + tile_size, k)

                A_tile = A[i:i_end, p:p_end]
                B_tile = B[p:p_end, j:j_end]

                # Multiply tiles and accumulate
                C[i:i_end, j:j_end] += np.matmul(A_tile, B_tile)

    return C


def benchmark_matrix_multiplication(m: int, k: int, n: int):
    """Benchmark different matrix multiplication implementations"""
    print(f"\n{'=' * 70}")
    print(f"Matrix Multiplication Benchmark")
    print(f"Matrix A: {m}x{k}, Matrix B: {k}x{n}, Result C: {m}x{n}")
    print(f"{'=' * 70}")

    # Initialize matrices
    A = np.random.rand(m, k).astype(np.float32)
    B = np.random.rand(k, n).astype(np.float32)

    print(f"\nMatrix A shape: {A.shape}")
    print(f"Matrix B shape: {B.shape}")
    print(f"Memory: A={A.nbytes/(1024*1024):.2f}MB, B={B.nbytes/(1024*1024):.2f}MB")
    print(f"\nSample from A (top-left 3x3):\n{A[:3, :3]}")
    print(f"\nSample from B (top-left 3x3):\n{B[:3, :3]}")

    results = []

    # Sequential CPU (only for small matrices)
    if m <= 256 and n <= 256:
        print(f"\n{'-' * 70}")
        print("Sequential CPU Matrix Multiplication (Naive)")
        print(f"{'-' * 70}")
        start = time.time()
        C_seq = matrix_multiply_cpu_sequential(A, B)
        time_seq = time.time() - start
        print(f"Time: {time_seq:.6f} seconds")
        print(f"Sample result (top-left 3x3):\n{C_seq[:3, :3]}")
        results.append(("Sequential CPU", time_seq, C_seq))
    else:
        print("\n(Skipping sequential naive implementation for large matrices)")
        C_seq = None
        time_seq = None

    # Optimized CPU (NumPy)
    print(f"\n{'-' * 70}")
    print("Optimized CPU Matrix Multiplication (NumPy)")
    print(f"{'-' * 70}")
    start = time.time()
    C_opt = matrix_multiply_cpu_optimized(A, B)
    time_opt = time.time() - start
    print(f"Time: {time_opt:.6f} seconds")
    print(f"Sample result (top-left 3x3):\n{C_opt[:3, :3]}")
    results.append(("Optimized CPU", time_opt, C_opt))

    # GPU Simulation (basic)
    if m <= 512 and n <= 512:
        print(f"\n{'-' * 70}")
        print("GPU-Style Matrix Multiplication (Simulated)")
        print(f"{'-' * 70}")
        block_size = 16
        start = time.time()
        C_gpu = simulate_gpu_matrix_multiply(A, B, block_size)
        time_gpu = time.time() - start
        grid_rows = (m + block_size - 1) // block_size
        grid_cols = (n + block_size - 1) // block_size
        print(f"Block size: {block_size}x{block_size}")
        print(f"Grid dimensions: {grid_rows}x{grid_cols} blocks")
        print(f"Total threads: {grid_rows * grid_cols * block_size * block_size:,}")
        print(f"Time: {time_gpu:.6f} seconds")
        print(f"Sample result (top-left 3x3):\n{C_gpu[:3, :3]}")
        results.append(("GPU Basic", time_gpu, C_gpu))

    # GPU Simulation (tiled)
    print(f"\n{'-' * 70}")
    print("GPU-Style Tiled Matrix Multiplication (Simulated)")
    print(f"{'-' * 70}")
    tile_size = 16
    start = time.time()
    C_gpu_tiled = simulate_gpu_matrix_multiply_tiled(A, B, tile_size)
    time_gpu_tiled = time.time() - start
    print(f"Tile size: {tile_size}x{tile_size}")
    print(f"Number of tiles: {(m*n) // (tile_size*tile_size):,}")
    print(f"Time: {time_gpu_tiled:.6f} seconds")
    print(f"Sample result (top-left 3x3):\n{C_gpu_tiled[:3, :3]}")
    results.append(("GPU Tiled", time_gpu_tiled, C_gpu_tiled))

    # Verification
    print(f"\n{'-' * 70}")
    print("Verification")
    print(f"{'-' * 70}")
    reference = C_opt
    for name, _, result in results:
        if name != "Optimized CPU":
            error = np.max(np.abs(reference - result))
            print(f"{name} max error: {error:.10f}")

    # Performance summary
    print(f"\n{'-' * 70}")
    print("Performance Summary")
    print(f"{'-' * 70}")
    print(f"{'Method':<35} {'Time (s)':<15} {'Speedup':<10}")
    print("-" * 70)

    base_time = time_opt
    for name, exec_time, _ in results:
        speedup = base_time / exec_time if exec_time > 0 else 0
        print(f"{name:<35} {exec_time:<15.6f} {speedup:<10.2f}x")


def explain_cuda_matrix_multiplication():
    """Explain CUDA matrix multiplication concepts"""
    print("\n" + "=" * 70)
    print("CUDA MATRIX MULTIPLICATION CONCEPTS")
    print("=" * 70)
    print("""
Matrix multiplication C = A × B is computationally intensive:
- For matrices of size NxN: O(N³) operations
- Perfect candidate for GPU parallelization

CUDA Approach:
1. Basic parallelization:
   - Each thread computes one element of C
   - Thread (i,j) computes C[i][j] = Σ(A[i][k] * B[k][j])

2. Tiled multiplication (optimized):
   - Uses shared memory for data reuse
   - Loads tiles of A and B into fast shared memory
   - Reduces global memory accesses significantly
   - Much better performance (3-10x speedup over basic)

Memory Hierarchy:
- Global Memory: Large but slow (~400-600 cycles latency)
- Shared Memory: Small but fast (~5 cycles latency)
- Registers: Fastest but very limited

Example for 16x16 block:
- Each block computes a 16x16 tile of C
- For 1024x1024 matrices: Need (1024/16)² = 4096 blocks
- Each block has 16×16 = 256 threads

CUDA C code structure:
__global__ void matMul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
    """)
    print("=" * 70)


def main():
    print("=" * 70)
    print("Assignment 4B: Matrix Multiplication")
    print("GPU Simulation using Python/NumPy")
    print("=" * 70)

    explain_cuda_matrix_multiplication()

    # Test with different sizes
    test_cases = [
        (256, 256, 256),   # Small matrices
        (512, 512, 512),   # Medium matrices
        (1024, 1024, 1024) # Large matrices
    ]

    for m, k, n in test_cases:
        benchmark_matrix_multiplication(m, k, n)
        print()

    print("\n" + "=" * 70)
    print("NOTE: For actual GPU acceleration, use:")
    print("  - CUDA C (see .cu reference files)")
    print("  - CuPy (CUDA-accelerated NumPy)")
    print("  - Numba with @cuda.jit decorator")
    print("  - PyTorch/TensorFlow for deep learning workloads")
    print("=" * 70)


if __name__ == "__main__":
    main()
