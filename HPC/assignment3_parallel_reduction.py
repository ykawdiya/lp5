"""
Assignment 3: Parallel Reduction (Min, Max, Sum, Average)
Simple implementation using parallel reduction concept
"""

import time
from multiprocessing import Pool, cpu_count


# ==================== SEQUENTIAL OPERATIONS ====================
def sequential_min(arr):
    """Sequential minimum"""
    return min(arr)


def sequential_max(arr):
    """Sequential maximum"""
    return max(arr)


def sequential_sum(arr):
    """Sequential sum"""
    return sum(arr)


def sequential_average(arr):
    """Sequential average"""
    return sum(arr) / len(arr)


# ==================== PARALLEL OPERATIONS ====================
def chunk_array(arr, num_chunks):
    """Divide array into chunks"""
    chunk_size = len(arr) // num_chunks
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(arr)
        chunks.append(arr[start:end])

    return chunks


def parallel_min(arr):
    """Parallel minimum using reduction"""
    chunks = chunk_array(arr, cpu_count())

    with Pool(cpu_count()) as pool:
        local_mins = pool.map(min, chunks)

    return min(local_mins)


def parallel_max(arr):
    """Parallel maximum using reduction"""
    chunks = chunk_array(arr, cpu_count())

    with Pool(cpu_count()) as pool:
        local_maxs = pool.map(max, chunks)

    return max(local_maxs)


def parallel_sum(arr):
    """Parallel sum using reduction"""
    chunks = chunk_array(arr, cpu_count())

    with Pool(cpu_count()) as pool:
        local_sums = pool.map(sum, chunks)

    return sum(local_sums)


def parallel_average(arr):
    """Parallel average using reduction"""
    total = parallel_sum(arr)
    return total / len(arr)


def benchmark(seq_func, par_func, arr, name):
    """Benchmark operation"""
    print(f"\n{name.upper()}")
    print("-" * 60)

    # Sequential
    start = time.time()
    seq_result = seq_func(arr)
    seq_time = time.time() - start
    print(f"Sequential: {seq_result:.6f} (Time: {seq_time:.6f}s)")

    # Parallel
    start = time.time()
    par_result = par_func(arr)
    par_time = time.time() - start
    print(f"Parallel:   {par_result:.6f} (Time: {par_time:.6f}s)")

    if seq_time > 0:
        print(f"Speedup: {seq_time / par_time:.2f}x")


def main():
    print("=" * 60)
    print("Assignment 3: Parallel Reduction Operations")
    print("=" * 60)

    # Test array
    arr = [5, 2, 9, 1, 7, 6, 8, 3, 4, 10] * 100

    print(f"\nArray size: {len(arr)}")
    print(f"First 10 elements: {arr[:10]}")
    print(f"CPU cores: {cpu_count()}")

    print("\nReduction Concept:")
    print("  1. Divide array into chunks (one per core)")
    print("  2. Each core computes local result")
    print("  3. Combine local results to get final result")

    # Test all operations
    operations = [
        (sequential_min, parallel_min, "Minimum"),
        (sequential_max, parallel_max, "Maximum"),
        (sequential_sum, parallel_sum, "Sum"),
        (sequential_average, parallel_average, "Average")
    ]

    for seq_func, par_func, name in operations:
        benchmark(seq_func, par_func, arr, name)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
