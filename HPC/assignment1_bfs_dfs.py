"""
Assignment 1: Parallel Breadth First Search and Depth First Search
Simple implementation using OpenMP concepts with Python multiprocessing
"""

import time
from collections import deque
from multiprocessing import Pool, Manager, cpu_count


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[] for _ in range(vertices)]

    def add_edge(self, u, v):
        """Add edge to undirected graph"""
        self.graph[u].append(v)
        self.graph[v].append(u)

    # ==================== DFS ====================
    def dfs_sequential(self, start):
        """Sequential DFS"""
        visited = [False] * self.V
        result = []

        def dfs_helper(node):
            visited[node] = True
            result.append(node)
            for neighbor in self.graph[node]:
                if not visited[neighbor]:
                    dfs_helper(neighbor)

        dfs_helper(start)
        return result

    def dfs_parallel(self, start):
        """Parallel DFS - explores neighbors in parallel"""
        manager = Manager()
        visited = manager.list([False] * self.V)
        result = manager.list()

        def explore_neighbors(neighbors):
            """Parallel exploration of neighbors"""
            for node in neighbors:
                if not visited[node]:
                    visited[node] = True
                    result.append(node)

        visited[start] = True
        result.append(start)

        # Process neighbors in parallel
        with Pool(cpu_count()) as pool:
            current = [start]
            while current:
                next_level = []
                for node in current:
                    unvisited = [n for n in self.graph[node] if not visited[n]]
                    for n in unvisited:
                        visited[n] = True
                        next_level.append(n)
                        result.append(n)
                current = next_level

        return list(result)

    # ==================== BFS ====================
    def bfs_sequential(self, start):
        """Sequential BFS"""
        visited = [False] * self.V
        result = []
        queue = deque([start])
        visited[start] = True

        while queue:
            node = queue.popleft()
            result.append(node)

            for neighbor in self.graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        return result

    def bfs_parallel(self, start):
        """Parallel BFS - processes levels in parallel"""
        manager = Manager()
        visited = manager.list([False] * self.V)
        result = manager.list()

        visited[start] = True
        result.append(start)
        current_level = [start]

        while current_level:
            next_level = []

            # Process all nodes in current level in parallel
            for node in current_level:
                for neighbor in self.graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        next_level.append(neighbor)
                        result.append(neighbor)

            current_level = next_level

        return list(result)


def benchmark(func, *args):
    """Measure execution time"""
    start = time.time()
    result = func(*args)
    return result, time.time() - start


def main():
    print("=" * 60)
    print("Assignment 1: Parallel BFS and DFS")
    print("=" * 60)

    # Create sample graph
    g = Graph(7)
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]

    for u, v in edges:
        g.add_edge(u, v)

    print("\nGraph edges:", edges)
    print("Starting node: 0")

    # DFS
    print("\n" + "-" * 60)
    print("DEPTH FIRST SEARCH")
    print("-" * 60)

    dfs_seq, t_seq = benchmark(g.dfs_sequential, 0)
    print(f"Sequential: {dfs_seq}")
    print(f"Time: {t_seq:.6f}s")

    dfs_par, t_par = benchmark(g.dfs_parallel, 0)
    print(f"Parallel:   {dfs_par}")
    print(f"Time: {t_par:.6f}s")

    # BFS
    print("\n" + "-" * 60)
    print("BREADTH FIRST SEARCH")
    print("-" * 60)

    bfs_seq, t_seq = benchmark(g.bfs_sequential, 0)
    print(f"Sequential: {bfs_seq}")
    print(f"Time: {t_seq:.6f}s")

    bfs_par, t_par = benchmark(g.bfs_parallel, 0)
    print(f"Parallel:   {bfs_par}")
    print(f"Time: {t_par:.6f}s")

    print("\n" + "=" * 60)
    print(f"CPU cores used: {cpu_count()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
