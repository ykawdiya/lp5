"""
Assignment 2: Parallel Bubble Sort and Merge Sort
Simple implementation with performance measurement
"""

import time
from multiprocessing import Pool, cpu_count


# ==================== BUBBLE SORT ====================
def bubble_sort_sequential(arr):
    """Sequential Bubble Sort"""
    arr = arr.copy()
    n = len(arr)

    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    return arr


def bubble_sort_parallel(arr):
    """Parallel Bubble Sort (Odd-Even Sort)"""
    arr = arr.copy()
    n = len(arr)

    for phase in range(n):
        if phase % 2 == 0:  # Even phase
            pairs = [(i, i + 1) for i in range(0, n - 1, 2)]
        else:  # Odd phase
            pairs = [(i, i + 1) for i in range(1, n - 1, 2)]

        # Compare and swap pairs
        for i, j in pairs:
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]

    return arr


# ==================== MERGE SORT ====================
def merge(left, right):
    """Merge two sorted arrays"""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort_sequential(arr):
    """Sequential Merge Sort"""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort_sequential(arr[:mid])
    right = merge_sort_sequential(arr[mid:])

    return merge(left, right)


def merge_sort_parallel(arr):
    """Parallel Merge Sort"""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2

    # Split and sort in parallel
    with Pool(2) as pool:
        left, right = pool.map(merge_sort_sequential, [arr[:mid], arr[mid:]])

    return merge(left, right)


def benchmark(func, arr, name):
    """Benchmark sorting function"""
    start = time.time()
    result = func(arr)
    elapsed = time.time() - start

    is_sorted = all(result[i] <= result[i + 1] for i in range(len(result) - 1))

    print(f"\n{name}:")
    print(f"  Time: {elapsed:.6f}s")
    print(f"  Sorted correctly: {is_sorted}")
    print(f"  Result: {result[:10]}...")

    return elapsed


def main():
    print("=" * 60)
    print("Assignment 2: Parallel Bubble Sort and Merge Sort")
    print("=" * 60)

    # Test array
    arr = [64, 34, 25, 12, 22, 11, 90, 88, 45, 50, 23, 36, 18, 77, 32]

    print(f"\nOriginal array ({len(arr)} elements):")
    print(arr)
    print(f"CPU cores: {cpu_count()}")

    # Bubble Sort
    print("\n" + "-" * 60)
    print("BUBBLE SORT")
    print("-" * 60)

    t_seq = benchmark(bubble_sort_sequential, arr, "Sequential")
    t_par = benchmark(bubble_sort_parallel, arr, "Parallel")

    if t_seq > 0:
        print(f"\nSpeedup: {t_seq / t_par:.2f}x")

    # Merge Sort
    print("\n" + "-" * 60)
    print("MERGE SORT")
    print("-" * 60)

    t_seq = benchmark(merge_sort_sequential, arr, "Sequential")
    t_par = benchmark(merge_sort_parallel, arr, "Parallel")

    if t_seq > 0:
        print(f"\nSpeedup: {t_seq / t_par:.2f}x")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
