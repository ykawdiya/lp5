# HPC Lab Assignments - Python Implementation

This directory contains Python implementations of all HPC lab assignments using parallel processing techniques.

## Assignments Overview

### Assignment 1: Parallel BFS and DFS (`assignment1_bfs_dfs.py`)
- **Topic**: Graph traversal algorithms with parallelization
- **Techniques**: Multiprocessing, parallel exploration of graph nodes
- **Run**: `python assignment1_bfs_dfs.py`

### Assignment 2: Parallel Sorting (`assignment2_sorting.py`)
- **Topic**: Bubble Sort and Merge Sort with parallel implementations
- **Techniques**: Odd-even transposition sort, parallel merge sort
- **Run**: `python assignment2_sorting.py`

### Assignment 3: Parallel Reduction (`assignment3_parallel_reduction.py`)
- **Topic**: Min, Max, Sum, Average operations using parallel reduction
- **Techniques**: Data partitioning, parallel aggregation
- **Run**: `python assignment3_parallel_reduction.py`

### Assignment 4A: Vector Addition (`assignment4a_vector_addition.py`)
- **Topic**: Large vector addition (GPU-style simulation)
- **Techniques**: NumPy vectorization, CUDA-style thread blocks simulation
- **Run**: `python assignment4a_vector_addition.py`

### Assignment 4B: Matrix Multiplication (`assignment4b_matrix_multiplication.py`)
- **Topic**: Matrix multiplication with tiling optimization
- **Techniques**: Tiled multiplication, shared memory simulation
- **Run**: `python assignment4b_matrix_multiplication.py`

### Assignment 5: ML/AI Application (`assignment5_ml_hpc.py`)
- **Topic**: Parallel neural network training with data parallelism
- **Techniques**: Distributed training, batch parallelization
- **Run**: `python assignment5_ml_hpc.py`

## Requirements

```bash
pip install numpy
```

No additional libraries required - uses Python standard library's `multiprocessing`.

## Running the Assignments

Each file is self-contained and can be run independently:

```bash
# Run any assignment
python assignment1_bfs_dfs.py
python assignment2_sorting.py
python assignment3_parallel_reduction.py
python assignment4a_vector_addition.py
python assignment4b_matrix_multiplication.py
python assignment5_ml_hpc.py
```

## Python vs C++ Implementation

The Python implementations use:
- **multiprocessing** → equivalent to OpenMP
- **NumPy** → vectorized operations for GPU-style parallelism
- **Pool workers** → parallel task execution

The C++ reference files use OpenMP and CUDA. Python provides similar parallelism patterns through:
- `Pool.map()` → parallel for loops
- `Manager.dict()` → shared memory
- NumPy → SIMD operations

## Performance Notes

- Python's multiprocessing has overhead (process creation)
- NumPy is highly optimized (uses C/Fortran under the hood)
- For production GPU work, use: CuPy, PyTorch, TensorFlow, or Numba
- Python is excellent for prototyping parallel algorithms

## Key Concepts Covered

1. **Parallelism Patterns**:
   - Task parallelism (independent operations)
   - Data parallelism (same operation on different data)
   - Pipeline parallelism (stages of computation)

2. **Performance Metrics**:
   - Speedup = Sequential Time / Parallel Time
   - Efficiency = Speedup / Number of Processors
   - Scalability (strong and weak)

3. **Synchronization**:
   - Critical sections
   - Atomic operations
   - Barriers

4. **Memory Models**:
   - Shared memory (threads)
   - Distributed memory (processes)
   - GPU memory hierarchy (global, shared, registers)

## Output

Each program provides:
- ✅ Performance comparison (sequential vs parallel)
- ✅ Timing measurements
- ✅ Speedup and efficiency metrics
- ✅ Correctness verification
- ✅ Conceptual explanations

## Tips for Understanding

1. Start with smaller data sizes to see clear output
2. Check CPU usage during execution (`htop` or Activity Monitor)
3. Compare the C++ reference for algorithmic understanding
4. Modify parameters (array sizes, workers) to see effects

## Viva Questions Reference

- What is the difference between parallelism and concurrency?
- Explain speedup and efficiency metrics
- Why is parallel reduction important?
- What is Amdahl's Law?
- Explain GPU memory hierarchy
- What is tiling/blocking in matrix multiplication?
- Difference between data and task parallelism?

---

**Note**: The `.cpp` and `.cu` files are reference implementations. The Python versions implement the same algorithms with equivalent parallel patterns.
