# HPC Assignments - Simplified Version

These are minimal implementations that satisfy the syllabus requirements. Each assignment is under 200 lines and focuses on core concepts.

## Assignments

### Assignment 1: Parallel BFS and DFS
**File:** `assignment1_bfs_dfs.py`

**What it does:**
- Implements BFS and DFS on a simple graph
- Sequential and parallel versions using multiprocessing
- Demonstrates tree/graph traversal parallelization

**Key concepts:**
- Graph representation using adjacency list
- DFS using recursion
- BFS using queue
- Parallel exploration of graph levels/neighbors

**Run:** `python assignment1_bfs_dfs.py`

---

### Assignment 2: Parallel Bubble Sort and Merge Sort
**File:** `assignment2_sorting.py`

**What it does:**
- Bubble Sort: Sequential and parallel (odd-even sort)
- Merge Sort: Sequential and parallel versions
- Performance comparison

**Key concepts:**
- Bubble sort with compare-swap operations
- Merge sort divide-and-conquer
- Parallel sorting using multiple processes
- Performance measurement

**Run:** `python assignment2_sorting.py`

---

### Assignment 3: Parallel Reduction
**File:** `assignment3_parallel_reduction.py`

**What it does:**
- Min, Max, Sum, Average operations
- Sequential vs parallel implementations
- Uses reduction pattern

**Key concepts:**
- Array chunking for parallel processing
- Reduction pattern (divide, compute, combine)
- Local vs global results
- Performance comparison

**Run:** `python assignment3_parallel_reduction.py`

---

### Assignment 5: HPC for AI/ML
**File:** `assignment5_ml_hpc.py`

**What it does:**
- Simple linear model training
- Data parallelism demonstration
- Sequential vs parallel training

**Key concepts:**
- Gradient descent training
- Batch processing
- Data parallelism
- Weight averaging across workers
- Model evaluation with MSE

**Run:** `python assignment5_ml_hpc.py`

---

## Notes

### Why is parallel slower?
For small problems, multiprocessing overhead (process creation, data serialization) dominates the speedup. Parallel algorithms show benefits with:
- Larger datasets (millions of elements)
- Complex computations per element
- Real HPC clusters with optimized communication

### Python vs OpenMP
- Python uses `multiprocessing` (similar to OpenMP concepts)
- OpenMP is used in C/C++ (see `.cpp` files for OpenMP versions)
- Both demonstrate the same parallel patterns

### For Practical Use
In production:
- Use NumPy/Pandas for array operations (optimized C backend)
- Use PyTorch/TensorFlow for ML (GPU acceleration)
- These assignments are educational demonstrations

---

## Syllabus Mapping

| Assignment | Syllabus Requirement |
|-----------|---------------------|
| 1 | Parallel BFS and DFS using OpenMP |
| 2 | Parallel Bubble Sort and Merge Sort using OpenMP |
| 3 | Min, Max, Sum, Average using Parallel Reduction |
| 5 | HPC application for AI/ML domain |

Assignment 4 (CUDA) uses GPU programming - see `.cu` files.
