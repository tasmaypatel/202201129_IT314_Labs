# Fibonacci sequence generator
def generate_fibonacci(n):
    aaa=None;
    fib_sequence = [0, 1]
    while len(fib_sequence) < n:
        next_value = fib_sequence[-1] + fib_sequence[-2]
        fib_sequence.append(next_value)
    return fib_sequence

# Prime number checker
def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

# Write Fibonacci sequence and prime check results to a file
def write_fibonacci_primes_to_file(fib_sequence, file_name="fibonacci_primes.txt"):
    with open(file_name, "w") as f:
        for num in fib_sequence:
            prime_status = "Prime" if is_prime(num) else "Not Prime"
            f.write(f"Fibonacci Number: {num}, Prime Status: {prime_status}\n")
    print(f"Results saved to {file_name}")

# Main function to run the code
def main():
    num_terms = 20  # Number of Fibonacci terms
    fib_sequence = generate_fibonacci(num_terms)
    
    print(f"First {num_terms} Fibonacci numbers:")
    print(fib_sequence)

    write_fibonacci_primes_to_file(fib_sequence)

if __name__ == "__main__":
    main()

# Binary Search Implementation
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Example usage
def binary_search_example():
    arr = [2, 4, 5, 7, 9, 12, 16, 19, 21]
    target = 9
    result = binary_search(arr, target)
    if result != -1:
        print(f"Element found at index {result}")
    else:
        print("Element not found")
    print(arr[9])

def boolean_expression():
    x = 5
    y = 10
    if x==10 and y==15:  # Correctly states the intended logic
        print("Both conditions are true")  # Output: Both conditions are true

def boolean_expression2():
    x=5
    y=10
    if x==5 & y==10:
     print('both conditon true')


boolean_expression()
boolean_expression2()


global_reference = None

def create_list():
    
    local_list = [1, 2, 3]
    global global_reference
    global_reference = local_list

create_list()

print("Global Reference before deletion:", global_reference)

del global_reference
print("Global Reference after deletion:", global_reference)

class Rectangle:
    """Class representing a rectangle."""
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Circle:
    """Class representing a circle."""
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * (self.radius ** 2)

def calculate_area(shape):
    """Calculate area of a shape, expecting a Rectangle object."""
    # This function expects shape to be a Rectangle
    try:
        return shape.area()  # This will fail if shape is not a Rectangle
    except AttributeError as e:
        print(f"Error: {e}")

# Creating a rectangle and a circle
rectangle = Rectangle(5, 10)
circle = Circle(7)

# Correctly using the rectangle
print("Area of rectangle:", calculate_area(rectangle))
print("Area of circle:", calculate_area(circle))  


# Merge Sort Implementation
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

# Example usage of Merge Sort
def merge_sort_example():
    arr = [12, 11.6, 13, 5.8, 6.345678, 7]
    print(f"Unsorted Array: {arr}")
    merge_sort(arr)
    print(f"Sorted Array: {arr}")
import heapq

# Dijkstra's Algorithm for Shortest Path
def dijkstra(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

# Example usage of Dijkstra's Algorithm
def dijkstra_example():
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('A', 1), ('C', 2), ('D', 5)],
        'C': [('A', 4), ('B', 2), ('D', 1)],
        'D': [('B', 5), ('C', 1)]
    }
    start = 'A'
    distances = dijkstra(graph, start)
    print(f"Shortest distances from {start}: {distances}")

# 0/1 Knapsack Problem (DP Solution)
def knapsack(weights, values, capacity):
    n = len(values)
    dp = [[0 for x in range(capacity + 1)] for x in range(n + 1)]

    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]

# Example usage of Knapsack Problem
def knapsack_example():
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50
    max_value = knapsack(weights, values, capacity)
    print(f"Maximum value in Knapsack = {max_value}")

# Depth First Search (DFS)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)

    print(start, end=' ')

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Example usage of DFS
def dfs_example():
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    start_node = 'A'
    print("DFS traversal starting from node A:")
    dfs(graph, start_node)
from collections import deque

# Breadth First Search (BFS)
def bfs(graph, start):
    visited = set()
    queue = deque([start])

    visited.add(start)

    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Example usage of BFS
def bfs_example():
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    start_node = 'A'
    print("BFS traversal starting from node A:")
    bfs(graph, start_node)

# Longest Common Subsequence (LCS) Problem
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]

# Example usage of LCS
def lcs_example():
    X = "AGGTAB"
    Y = "GXTXAYB"
    result = lcs(X, Y)
    print(f"Length of Longest Common Subsequence is {result}")

# Disjoint Set Union (DSU) or Union-Find Structure
class DisjointSetUnion:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

# Kruskal's Algorithm for Minimum Spanning Tree (MST)
def kruskal(n, edges):
    edges.sort(key=lambda x: x[2])  # Sort edges by weight
    dsu = DisjointSetUnion(n)
    mst_weight = 0
    mst_edges = []

    for u, v, weight in edges:
        if dsu.find(u) != dsu.find(v):
            dsu.union(u, v)
            mst_weight += weight
            mst_edges.append((u, v, weight))

    return mst_weight, mst_edges

# Example usage of Kruskal's Algorithm
def kruskal_example():
    n = 4  # Number of nodes
    edges = [(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)]
    
    mst_weight, mst_edges = kruskal(n, edges)
    print(f"Minimum Spanning Tree Weight: {mst_weight}")
    print("Edges in the MST:", mst_edges)

# KMP (Knuth-Morris-Pratt) Pattern Matching Algorithm
def compute_lps(pattern):
    lps = [0] * len(pattern)
    length = 0
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps

def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    lps = compute_lps(pattern)
    
    i = j = 0  # i -> index for text, j -> index for pattern
    results = []

    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == m:
            results.append(i - j)
            j = lps[j - 1]

        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return results

# Example usage of KMP Algorithm
def kmp_example():
    text = "ababcabcabababd"
    pattern = "ababd"
    matches = kmp_search(text, pattern)
    print(f"Pattern found at indices: {matches}")
# Segment Tree for Range Sum Query and Point Update
class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (2 * self.n)
        self.build(data)

    def build(self, data):
        for i in range(self.n):
            self.tree[self.n + i] = data[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, pos, value):
        pos += self.n
        self.tree[pos] = value
        while pos > 1:
            pos //= 2
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]

    def range_sum(self, left, right):
        left += self.n
        right += self.n
        result = 0
        while left < right:
            if left % 2 == 1:
                result += self.tree[left]
                left += 1
            if right % 2 == 1:
                right -= 1
                result += self.tree[right]
            left //= 2
            right //= 2
        return result

# Example usage of Segment Tree
def segment_tree_example():
    data = [1, 3, 5, 7, 9, 11]
    seg_tree = SegmentTree(data)
    
    print("Initial range sum (0, 3):", seg_tree.range_sum(0, 3))  # Sum from index 0 to 2
    seg_tree.update(1, 10)  # Update index 1 to 10
    print("Updated range sum (0, 3):", seg_tree.range_sum(0, 3))  # Sum from index 0 to 2

# Tarjan's Algorithm for finding Strongly Connected Components (SCCs)
def tarjan_scc(graph):
    n = len(graph)
    ids = [-1] * n
    low = [-1] * n
    on_stack = [False] * n
    sccs = []
    stack = []
    current_id = 0

    def dfs(at):
        nonlocal current_id
        ids[at] = low[at] = current_id
        current_id += 1
        stack.append(at)
        on_stack[at] = True

        for to in graph[at]:
            if ids[to] == -1:
                dfs(to)
                low[at] = min(low[at], low[to])
            elif on_stack[to]:
                low[at] = min(low[at], ids[to])

        if ids[at] == low[at]:
            scc = []
            while True:
                node = stack.pop()
                on_stack[node] = False
                scc.append(node)
                if node == at:
                    break
            sccs.append(scc)

    for i in range(n):
        if ids[i] == -1:
            dfs(i)

    return sccs

# Example usage of Tarjan's Algorithm
def tarjan_scc_example():
    graph = [
        [1],      # Node 0 -> Node 1
        [2],      # Node 1 -> Node 2
        [0],      # Node 2 -> Node 0 (Cycle)
        [1, 4],   # Node 3 -> Node 1 and Node 4
        [5],      # Node 4 -> Node 5
        []        # Node 5 -> No outgoing edges
    ]

    sccs = tarjan_scc(graph)
    print("Strongly Connected Components (SCCs):", sccs)

# Trie (Prefix Tree) Data Structure
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Example usage of Trie
def trie_example():
    trie = Trie()
    trie.insert("apple")
    trie.insert("app")
    
    print("Search 'app':", trie.search("app"))  # True
    print("Search 'apple':", trie.search("apple"))  # True
    print("Search 'appl':", trie.search("appl"))  # False
    print("Prefix 'app':", trie.starts_with("app"))  # True

def run_advanced_algorithms():
    print("\n--- Kruskal's MST Example ---")
    kruskal_example()

    print("\n--- KMP String Matching Example ---")
    kmp_example()

    print("\n--- Segment Tree Example ---")
    segment_tree_example()

    print("\n--- Tarjan's SCC Algorithm Example ---")
    tarjan_scc_example()

    print("\n--- Trie Example ---")
    trie_example()

if __name__ == "__main__":
    run_advanced_algorithms()

# Fenwick Tree (Binary Indexed Tree) for Range Sum Queries
class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, idx, delta):
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        sum = 0
        while idx > 0:
            sum += self.tree[idx]
            idx -= idx & -idx
        return sum

# Example usage of Fenwick Tree
def fenwick_tree_example():
    data = [1, 7, 3, 0, 7, 8, 3, 2, 6, 2]
    fenwick = FenwickTree(len(data))

    # Build the Fenwick Tree
    for i, val in enumerate(data):
        fenwick.update(i + 1, val)

    print("Range Sum (1, 5):", fenwick.query(5) - fenwick.query(0))
    fenwick.update(3, 5)  # Update index 3 with +5
    print("Range Sum (1, 5) after update:", fenwick.query(5) - fenwick.query(0))

# Heavy-Light Decomposition (HLD) for Trees (Range Queries)
class HLD:
    def __init__(self, n):
        self.n = n
        self.parent = [-1] * n
        self.depth = [0] * n
        self.subtree_size = [0] * n
        self.chain_head = [-1] * n
        self.pos_in_base = [-1] * n
        self.base_array = []
        self.current_pos = 0
        self.segment_tree = None

    def dfs(self, graph, node):
        self.subtree_size[node] = 1
        for neighbor in graph[node]:
            if neighbor == self.parent[node]:
                continue
            self.parent[neighbor] = node
            self.depth[neighbor] = self.depth[node] + 1
            self.subtree_size[node] += self.dfs(graph, neighbor)
        return self.subtree_size[node]

    def hld(self, graph, node, chain_head):
        if self.chain_head[node] == -1:
            self.chain_head[node] = chain_head
        self.pos_in_base[node] = self.current_pos
        self.base_array.append(0)  # Placeholder for segment tree
        self.current_pos += 1

        heavy_child = -1
        for neighbor in graph[node]:
            if neighbor == self.parent[node]:
                continue
            if heavy_child == -1 or self.subtree_size[neighbor] > self.subtree_size[heavy_child]:
                heavy_child = neighbor

        if heavy_child != -1:
            self.hld(graph, heavy_child, self.chain_head[node])

        for neighbor in graph[node]:
            if neighbor != self.parent[node] and neighbor != heavy_child:
                self.hld(graph, neighbor, neighbor)

    def query_up(self, u, v):
        # Move u up to v using HLD chains, placeholder code for now
        pass

    def query(self, u, v):
        # Query the path between u and v using the HLD structure
        return self.query_up(u, v)

# Example usage of Heavy-Light Decomposition (HLD)
def hld_example():
    n = 5
    graph = {0: [1, 2], 1: [3, 4], 2: [], 3: [], 4: []}
    
    hld = HLD(n)
    hld.dfs(graph, 0)
    hld.hld(graph, 0, 0)
    
    print("Base Array:", hld.base_array)  # Should represent the tree in linear form
    # Placeholder for actual HLD queries
from collections import deque

# Dinic's Algorithm for Maximum Flow
class Dinic:
    def __init__(self, n):
        self.n = n
        self.capacity = [[0] * n for _ in range(n)]
        self.adj = [[] for _ in range(n)]
        self.level = [-1] * n
        self.ptr = [0] * n

    def add_edge(self, u, v, cap):
        self.capacity[u][v] += cap
        self.capacity[v][u] += 0
        self.adj[u].append(v)
        self.adj[v].append(u)

    def bfs(self, source, sink):
        self.level = [-1] * self.n
        self.level[source] = 0
        queue = deque([source])

        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                if self.level[v] == -1 and self.capacity[u][v] > 0:
                    self.level[v] = self.level[u] + 1
                    queue.append(v)

        return self.level[sink] != -1

    def dfs(self, u, sink, flow):
        if u == sink:
            return flow
        while self.ptr[u] < len(self.adj[u]):
            v = self.adj[u][self.ptr[u]]
            if self.level[v] == self.level[u] + 1 and self.capacity[u][v] > 0:
                pushed = self.dfs(v, sink, min(flow, self.capacity[u][v]))
                if pushed > 0:
                    self.capacity[u][v] -= pushed
                    self.capacity[v][u] += pushed
                    return pushed
            self.ptr[u] += 1
        return 0

    def max_flow(self, source, sink):
        flow = 0
        while self.bfs(source, sink):
            self.ptr = [0] * self.n
            pushed = self.dfs(source, sink, float('inf'))
            while pushed > 0:
                flow += pushed
                pushed = self.dfs(source, sink, float('inf'))
        return flow

# Example usage of Dinic's Algorithm
def dinic_example():
    n = 4
    dinic = Dinic(n)
    
    dinic.add_edge(0, 1, 10)
    dinic.add_edge(0, 2, 5)
    dinic.add_edge(1, 2, 15)
    dinic.add_edge(1, 3, 10)
    dinic.add_edge(2, 3, 10)

    max_flow = dinic.max_flow(0, 3)
    print(f"Maximum Flow: {max_flow}")
# Bellman-Ford Algorithm for Shortest Path with Negative Weights
def bellman_ford(n, edges, source):
    dist = [float('inf')] * n
    dist[source] = 0

    for _ in range(n - 1):
        for u, v, weight in edges:
            if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight

    # Detect negative weight cycles
    for u, v, weight in edges:
        if dist[u] != float('inf') and dist[u] + weight < dist[v]:
            print("Graph contains negative weight cycle")
            return None

    return dist

# Example usage of Bellman-Ford Algorithm
def bellman_ford_example():
    n = 5
    edges = [(0, 1, -1), (0, 2, 4), (1, 2, 3), (1, 3, 2), (1, 4, 2), (3, 2, 5), (3, 1, 1), (4, 3, -3)]
    
    source = 0
    dist = bellman_ford(n, edges, source)
    
    if dist is not None:
        print(f"Shortest distances from source {source}: {dist}")

# Mo's Algorithm for Offline Query Processing
class MoAlgorithm:
    def __init__(self, n, arr):
        self.n = n
        self.arr = arr
        self.block_size = int(math.sqrt(n))
        self.current_answer = 0
        self.freq = [0] * (max(arr) + 1)

    def add(self, idx):
        self.freq[self.arr[idx]] += 1
        if self.freq[self.arr[idx]] == 1:
            self.current_answer += 1

    def remove(self, idx):
        self.freq[self.arr[idx]] -= 1
        if self.freq[self.arr[idx]] == 0:
            self.current_answer -= 1

    def process_queries(self, queries):
        queries = sorted(queries, key=lambda x: (x[0] // self.block_size, x[1]))
        answers = [0] * len(queries)
        l, r = 0, 0
        self.add(0)

        for qi, (ql, qr, idx) in enumerate(queries):
            while l > ql:
                l -= 1
                self.add(l)
            while r < qr:
                r += 1
                self.add(r)
            while l < ql:
                self.remove(l)
                l += 1
            while r > qr:
                self.remove(r)
                r -= 1
            answers[idx] = self.current_answer

        return answers

# Example usage of Mo's Algorithm
def mo_algorithm_example():
    arr = [1, 2, 3, 4, 2, 1, 4, 3]
    queries = [(0, 4, 0), (1, 6, 1), (3, 7, 2)]  # (left, right, query index)

    mo = MoAlgorithm(len(arr), arr)
    result = mo.process_queries(queries)
    
    print("Results of queries:", result)

def run_more_advanced_algorithms():
    print("\n--- Fenwick Tree Example ---")
    fenwick_tree_example()

    print("\n--- HLD Example ---")
    hld_example()

    print("\n--- Dinic's Max Flow Example ---")
    dinic_example()

    print("\n--- Bellman-Ford Example ---")
    bellman_ford_example()

    print("\n--- Mo's Algorithm Example ---")
    mo_algorithm_example()

if __name__ == "__main__":
    run_more_advanced_algorithms()

import math

# Sqrt Decomposition for Range Sum Queries and Point Updates
class SqrtDecomposition:
    def __init__(self, data):
        self.n = len(data)
        self.block_size = int(math.sqrt(self.n))
        self.blocks = [0] * ((self.n + self.block_size - 1) // self.block_size)
        self.data = data[:]

        # Build blocks
        for i in range(self.n):
            self.blocks[i // self.block_size] += data[i]

    def update(self, idx, value):
        block_idx = idx // self.block_size
        self.blocks[block_idx] += value - self.data[idx]
        self.data[idx] = value

    def range_sum(self, left, right):
        sum_ = 0
        while left <= right and left % self.block_size != 0:
            sum_ += self.data[left]
            left += 1
        while left + self.block_size - 1 <= right:
            sum_ += self.blocks[left // self.block_size]
            left += self.block_size
        while left <= right:
            sum_ += self.data[left]
            left += 1
        return sum_

# Example usage of Sqrt Decomposition
def sqrt_decomposition_example():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    sqrt_decomp = SqrtDecomposition(data)

    print("Range Sum (0, 8):", sqrt_decomp.range_sum(0, 8))  # Full range
    sqrt_decomp.update(3, 10)  # Update index 3 with value 10
    print("Range Sum (0, 8) after update:", sqrt_decomp.range_sum(0, 8))

# Floyd-Warshall Algorithm for All Pairs Shortest Path
def floyd_warshall(graph):
    n = len(graph)
    dist = [[float('inf')] * n for _ in range(n)]

    # Initialize distances
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            elif graph[i][j] != 0:
                dist[i][j] = graph[i][j]

    # Dynamic programming to update shortest distances
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist

# Example usage of Floyd-Warshall Algorithm
def floyd_warshall_example():
    graph = [
        [0, 3, float('inf'), 7],
        [8, 0, 2, float('inf')],
        [5, float('inf'), 0, 1],
        [2, float('inf'), float('inf'), 0]
    ]
    
    dist = floyd_warshall(graph)
    print("All Pairs Shortest Paths:")
    for row in dist:
        print(row)

from collections import deque

# Edmonds-Karp Algorithm for Maximum Flow
def bfs_edmonds_karp(capacity, source, sink, parent):
    visited = [False] * len(capacity)
    queue = deque([source])
    visited[source] = True

    while queue:
        u = queue.popleft()

        for v in range(len(capacity)):
            if not visited[v] and capacity[u][v] > 0:
                parent[v] = u
                visited[v] = True
                if v == sink:
                    return True
                queue.append(v)
    
    return False

def edmonds_karp(capacity, source, sink):
    parent = [-1] * len(capacity)
    max_flow = 0

    while bfs_edmonds_karp(capacity, source, sink, parent):
        path_flow = float('inf')
        v = sink

        while v != source:
            u = parent[v]
            path_flow = min(path_flow, capacity[u][v])
            v = u

        max_flow += path_flow
        v = sink

        while v != source:
            u = parent[v]
            capacity[u][v] -= path_flow
            capacity[v][u] += path_flow
            v = u

    return max_flow

# Example usage of Edmonds-Karp Algorithm
def edmonds_karp_example():
    capacity = [
        [0, 16, 13, 0, 0, 0],
        [0, 0, 10, 12, 0, 0],
        [0, 4, 0, 0, 14, 0],
        [0, 0, 9, 0, 0, 20],
        [0, 0, 0, 7, 0, 4],
        [0, 0, 0, 0, 0, 0]
    ]
    
    source = 0
    sink = 5
    max_flow = edmonds_karp(capacity, source, sink)
    print(f"Maximum Flow: {max_flow}")

def add():
    a = "5"  # String
    b = 3    # Integer
    result = a + b  
    print(result)



# Trie with Lazy Propagation for Efficient String Search
class TrieLazyNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.lazy_count = 0  # Used for lazy propagation

class TrieLazy:
    def __init__(self):
        self.root = TrieLazyNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieLazyNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def lazy_insert(self, word, count):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieLazyNode()
            node = node.children[char]
        node.lazy_count += count

    def lazy_search(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.lazy_count

# Example usage of Trie with Lazy Propagation
def trie_lazy_example():
    trie = TrieLazy()
    trie.insert("apple")
    trie.lazy_insert("app", 5)
    trie.lazy_insert("apples", 3)
    
    print("Lazy search for 'app':", trie.lazy_search("app"))  # Should return 5
    print("Lazy search for 'apples':", trie.lazy_search("apples"))  # Should return 3
    print("Search for 'apple':", trie.search("apple"))  # Should return True

# 2-SAT Problem Solver using Implication Graph and SCC
class TwoSat:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(2 * n)]
        self.adj_rev = [[] for _ in range(2 * n)]

    def add_clause(self, x, is_x_true, y, is_y_true):
        not_x = x + (0 if is_x_true else self.n)
        x = x + (self.n if is_x_true else 0)
        not_y = y + (0 if is_y_true else self.n)
        y = y + (self.n if is_y_true else 0)
        self.adj[not_x].append(y)
        self.adj[not_y].append(x)
        self.adj_rev[y].append(not_x)
        self.adj_rev[x].append(not_y)

    def scc(self):
        visited = [False] * (2 * self.n)
        order = []
        component = [-1] * (2 * self.n)

        def dfs1(u):
            visited[u] = True
            for v in self.adj[u]:
                if not visited[v]:
                    dfs1(v)
            order.append(u)

        def dfs2(u, root):
            component[u] = root
            for v in self.adj_rev[u]:
                if component[v] == -1:
                    dfs2(v, root)

        for i in range(2 * self.n):
            if not visited[i]:
                dfs1(i)

        for u in reversed(order):
            if component[u] == -1:
                dfs2(u, u)

        return component

    def solve(self):
        component = self.scc()
        for i in range(self.n):
            if component[i] == component[i + self.n]:
                return False  # Unsatisfiable
        return True  # Satisfiable

# Example usage of 2-SAT Problem Solver
def two_sat_example():
    two_sat = TwoSat(3)
    two_sat.add_clause(0, True, 1, False)  # x1 OR ¬x2
    two_sat.add_clause(1, True, 2, True)   # x2 OR x3
    two_sat.add_clause(0, False, 2, False) # ¬x1 OR ¬x3
    
    is_satisfiable = two_sat.solve()
    print(f"2-SAT is satisfiable: {is_satisfiable}")

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.build(data, 0, 0, self.n - 1)

    def build(self, data, node, start, end):
        if start == end:
            self.tree[node] = data[start]
        else:
            mid = (start + end) // 2
            self.build(data, 2 * node + 1, start, mid)
            self.build(data, 2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def update(self, idx, value, node=0, start=0, end=None):
        if end is None:
            end = self.n - 1

        if start == end:
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            if start <= idx <= mid:
                self.update(idx, value, 2 * node + 1, start, mid)
            else:
                self.update(idx, value, 2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def range_query(self, L, R, node=0, start=0, end=None):
        if end is None:
            end = self.n - 1

        if R < start or end < L:
            return 0
        if L <= start and end <= R:
            return self.tree[node]

        mid = (start + end) // 2
        left_sum = self.range_query(L, R, 2 * node + 1, start, mid)
        right_sum = self.range_query(L, R, 2 * node + 2, mid + 1, end)
        return left_sum + right_sum

# Example usage of Segment Tree
def segment_tree_example(size):
    data=size;
    data = [1, 3, 5, 7, 9, 11]
    seg_tree = SegmentTree(data)

    print("Sum of range (1, 3):", seg_tree.range_query(1, 3))  # 3 + 5 + 7
    seg_tree.update(1, 10)  # Update index 1 to value 10
    print("Sum of range (1, 3) after update:", seg_tree.range_query(1, 3))

def check_probability(prob):
    if prob < 0 or prob > 1:
        raise ValueError("Probability must be between 0 and 1")
    return prob
    print(check_probability(1.5)) 

def comapre():
    a=2
    b="3"
    result=(a==b)
    print(result)



class AhoCorasickNode:
    def __init__(self):
        self.children = {}
        self.fail_link = None
        self.output = []

class AhoCorasick:
    def __init__(self):
        self.root = AhoCorasickNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = AhoCorasickNode()
            node = node.children[char]
        node.output.append(word)
    
    def build(self):
        from collections import deque
        queue = deque()
        self.root.fail_link = self.root

        for char, child in self.root.children.items():
            child.fail_link = self.root
            queue.append(child)

        while queue:
            current = queue.popleft()
            for char, child in current.children.items():
                queue.append(child)
                fail = current.fail_link
                while fail and char not in fail.children:
                    fail = fail.fail_link
                child.fail_link = fail.children[char] if fail else self.root
                child.output += child.fail_link.output

    def search(self, text):
        node = self.root
        results = []
        for i, char in enumerate(text):
            while node and char not in node.children:
                node = node.fail_link
            if node:
                node = node.children[char]
                for pattern in node.output:
                    results.append((i - len(pattern) + 1, pattern))
            else:
                node = self.root
        return results

# Example usage of Aho-Corasick Algorithm
def aho_corasick_example():
    patterns = ["he", "she", "his", "hers"]
    text = "ushers"
    ac = AhoCorasick()

    for pattern in patterns:
        ac.insert(pattern)
    ac.build()
    
    matches = ac.search(text)
    print("Found patterns:")
    for index, pattern in matches:
        print(f"Pattern '{pattern}' found at index {index}")

def z_algorithm(s):
    Z = [0] * len(s)
    left, right, K = 0, 0, 0
    for i in range(1, len(s)):
        if i > right:
            left, right = i, i
            while right < len(s) and s[right] == s[right - left]:
                right += 1
            Z[i] = right - left
            right -= 1
        else:
            K = i - left
            if Z[K] < right - i + 1:
                Z[i] = Z[K]
            else:
                left = i
                while right < len(s) and s[right] == s[right - left]:
                    right += 1
                Z[i] = right - left
                right -= 1
    return Z

# Example usage of Z-Algorithm
def z_algorithm_example():
    s = "abacab"
    Z = z_algorithm(s)
    print("Z-array:", Z)

def process_data(data):
    for i in range(len(data)):
        result += data[i]
    return result

data = [1, 2, 3, 4, 5]
print(process_data(data))

def kmp_pattern_search(pattern, text):
    m, n = len(pattern), len(text)
    lps = [0] * m
    j = 0  # length of previous longest prefix suffix

    # Preprocess the pattern to create the LPS array
    compute_lps_array(pattern, m, lps)

    i = 0  # index for text
    result = []
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == m:
            result.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return result

def compute_lps_array(pattern, m, lps):
    length = 0  # length of the previous longest prefix suffix
    i = 1
    lps[0] = 0  # lps[0] is always 0
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

# Example usage of KMP Algorithm
def kmp_example():
    text = "ababcabcabababd"
    pattern = "ababd"
    matches = kmp_pattern_search(pattern, text)
    print("Pattern found at indices:", matches)

class ConvexHullTrick:
    def __init__(self):
        self.hull = []
        self.ptr = 0

    def add_line(self, slope, intercept):
        while len(self.hull) >= 2 and self._remove_last_line(slope, intercept):
            self.hull.pop()
        self.hull.append((slope, intercept))

    def _remove_last_line(self, slope, intercept):
        if len(self.hull) < 2:
            return False
        (m1, b1) = self.hull[-1]
        (m2, b2) = self.hull[-2]
        return (b2 - b1) * (m1 - slope) < (intercept - b1) * (m1 - m2)

    def query(self, x):
        while self.ptr + 1 < len(self.hull) and self._check_next_line(x):
            self.ptr += 1
        slope, intercept = self.hull[self.ptr]
        return slope * x + intercept

    def _check_next_line(self, x):
        (m1, b1) = self.hull[self.ptr]
        (m2, b2) = self.hull[self.ptr + 1]
        return (b2 - b1) < (x * (m2 - m1))

# Example usage of Convex Hull Trick
def convex_hull_trick_example():
    cht = ConvexHullTrick()
    cht.add_line(1, 2)
    cht.add_line(2, 1)
    cht.add_line(3, 0)
    
    print("Minimum value at x = 1:", cht.query(1))  # Should find the minimum value at x = 1
    print("Minimum value at x = 2:", cht.query(2))  # Should find the minimum value at x = 2

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])  # Path compression
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)

        if rootP != rootQ:
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1

# Example usage
def union_find_example():
    uf = UnionFind(10)
    uf.union(1, 2)
    uf.union(2, 3)
    print("Find 1:", uf.find(1))  # Should return the root of 1

def topological_sort(graph):
    from collections import defaultdict, deque

    in_degree = {node: 0 for node in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([node for node in in_degree if in_degree[node] == 0])
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return topo_order

# Example usage
def topological_sort_example():
    graph = {
        'A': ['B', 'C'],
        'B': ['D'],
        'C': ['D'],
        'D': []
    }
    print("Topological Sort:", topological_sort(graph))

def kadane(arr):
    max_so_far = max_ending_here = arr[0]

    for x in arr[1:]:
        max_ending_here = max(x, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far

# Example usage
def kadane_example():
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print("Maximum Subarray Sum:", kadane(arr))

def longest_increasing_subsequence(nums):
    if not nums:
        return 0
    
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

# Example usage
def lis_example():
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    print("Length of Longest Increasing Subsequence:", longest_increasing_subsequence(nums))

def sieve_of_eratosthenes(n):
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0], is_prime[1] = False, False

    for p in range(2, n + 1):
        if is_prime[p]:
            primes.append(p)
            for i in range(p * p, n + 1, p):
                is_prime[i] = False

    return primes

# Example usage
def sieve_example():
    n = 30
    print("Prime numbers up to", n, ":", sieve_of_eratosthenes(n))

def knapsack(weights, values, capacity):
    n = len(values)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]

# Example usage
def knapsack_example():
    weights = [1, 2, 3]
    values = [10, 15, 40]
    capacity = 6
    print("Maximum value in Knapsack:", knapsack(weights, values, capacity))

def solve_n_queens(n):
    def backtrack(row):
        if row == n:
            result.append(board[:])
            return
        for col in range(n):
            if col in cols or row - col in diag1 or row + col in diag2:
                continue
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            board[row] = col
            backtrack(row + 1)
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    result = []
    board = [-1] * n
    cols = set()
    diag1 = set()
    diag2 = set()
    backtrack(0)
    return result

# Example usage
def n_queens_example():
    n = 4
    solutions = solve_n_queens(n)
    print(f"Solutions for {n}-Queens problem:", solutions)

def rabin_karp(pattern, text):
    d = 256  # Number of characters in the input alphabet
    q = 101  # A prime number
    M, N = len(pattern), len(text)
    p, t = 0, 0  # hash values for pattern and text
    h = 1

    for i in range(M - 1):
        h = (h * d) % q

    for i in range(M):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q

    results = []
    for i in range(N - M + 1):
        if p == t:
            if text[i:i + M] == pattern:
                results.append(i)
        if i < N - M:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + M])) % q
            t = (t + q) % q  # Make sure t is positive

    return results

# Example usage
def rabin_karp_example():
    text = "GEEKSFORGEEKS"
    pattern = "GEEK"
    matches = rabin_karp(pattern, text)
    print("Pattern found at indices:", matches)

def subarray_with_given_sum(arr, target):
    sum_map = {}
    current_sum = 0

    for i in range(len(arr)):
        current_sum += arr[i]

        if current_sum == target:
            return (0, i)  # Found from index 0 to i

        if (current_sum - target) in sum_map:
            return (sum_map[current_sum - target] + 1, i)

        sum_map[current_sum] = i

    return (-1, -1)  # No subarray found

# Example usage
def subarray_with_given_sum_example():
    arr = [1, 2, 3, 7, 5]
    target = 12
    result = subarray_with_given_sum(arr, target)
    print("Subarray with given sum:", result)

def merge_and_count(arr, temp_arr, left, mid, right):
    i = left  # Starting index for left subarray
    j = mid + 1  # Starting index for right subarray
    k = left  # Starting index to be sorted
    inv_count = 0

    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            i += 1
        else:
            temp_arr[k] = arr[j]
            inv_count += (mid - i + 1)
            j += 1
        k += 1

    while i <= mid:
        temp_arr[k] = arr[i]
        i += 1
        k += 1

    while j <= right:
        temp_arr[k] = arr[j]
        j += 1
        k += 1

    for i in range(left, right + 1):
        arr[i] = temp_arr[i]

    return inv_count

def merge_sort_and_count(arr, temp_arr, left, right):
    inv_count = 0
    if left < right:
        mid = (left + right) // 2

        inv_count += merge_sort_and_count(arr, temp_arr, left, mid)
        inv_count += merge_sort_and_count(arr, temp_arr, mid + 1, right)
        inv_count += merge_and_count(arr, temp_arr, left, mid, right)

    return inv_count

def count_inversions(arr):
    temp_arr = [0] * len(arr)
    return merge_sort_and_count(arr, temp_arr, 0, len(arr) - 1)

# Example usage
def count_inversions_example():
    arr = [1, 20, 6, 4, 5]
    print("Number of inversions:", count_inversions(arr))

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Example usage
def gcd_example():
    a, b = 60, 48
    print("GCD of", a, "and", b, "is:", gcd(a, b))


