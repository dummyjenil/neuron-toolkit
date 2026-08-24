# Component Flows & Algorithmic Specifications

This document defines the exact step-by-step technical flows, algorithms, data structures, and runtime complexity characteristics of all core components in `neuron-toolkit`.

---

## 1. Model Loading & Backend Parsing Flow

### Flow Specifications:

1. **Source Inspection**: Checks whether the input source is a `str` path, `bytes` buffer, ONNX `ModelProto`, or TFLite `Model`.
2. **Magic Byte Verification**: For `bytes` buffers, offset `4:8` is inspected for the byte string `b"TFL3"`.
3. **Shape Inference Pipeline**:
   - For ONNX: Executes `onnx.shape_inference.infer_shapes()`. Extracts tensor type annotations from `graph.value_info`, `graph.input`, and `graph.output`.
   - For TFLite: Iterates over `subgraph.Tensors()`, building a mapping of `(rank, numpy_dtype_string)`.
4. **Lazy Tensor Mapping**: Wraps initializers in `LazyTensorMap` so array buffers are deserialized via `numpy_helper.to_array()` or `buffer.DataAsNumpy()` on-demand.

### Time & Memory Complexity:

- **Parse Time**: $O(V + E)$ where $V$ is operator count and $E$ is tensor wire count.
- **Memory Overhead**: $O(V)$ for node structures. Initializer weight arrays remain zero-copy / un-instantiated until indexed.

---

## 2. Query Traversal & Graph Search Flow (`NeuronQuery`)

### Flow Specifications:

1. **Initialization**: Stores `nodes`, `tensor_map`, `all_nodes`, `graph_inputs`, `graph_outputs`, `shape_info`, and optional `backend`.
2. **Cached Index Construction**:
   - `output_map`: Hash map $O(V)$ indexing output tensor names to producer nodes.
   - `_nx_graph`: NetworkX `DiGraph` representation of operator connectivity.
   - `_node_to_idx`: Kahn's / DFS topological ordering hash map.
3. **Filter Chain Execution**:
   - Predicates like `.op()`, `.name()`, `.attr()`, `.rank()`, `.dtype()` filter the internal `self.nodes` list.
   - Returns a cloned `NeuronQuery` sharing the cached `output_map`, `_nx_graph`, and `_node_to_idx`.
4. **Multi-Source BFS Traversal (`ancestors` & `descendants`)**:
   - Uses queue-based multi-source BFS over `_nx_graph` or reversed `_nx_graph`.
   - Bounded by `max_depth`.

### Complexity:

- **Filtering Time**: $O(K)$ where $K$ is currently selected node count.
- **Traversal Time**: $O(V_{sub} + E_{sub})$ where $V_{sub}$ and $E_{sub}$ are connected sub-components.

---

## 3. Recursive DFS Pattern Matcher Flow (`PatternDetector`)

### Flow Specifications:

1. **Target Candidate Scan**: Iterates over all reachable nodes or descendant candidates of `start_node`.
2. **`MatchContext` State Management**:
   - Tracks `bindings` (capture dict), `trail` (node list), `memo` (pattern identity map), `visited` (node names).
   - Snapshot & restore mechanism for backtracking on failure.
3. **Referential Consistency**:
   - `pat_id = id(pattern)`: If `pat_id` exists in `ctx.memo`, verifies that `ctx.memo[pat_id] == id(node)`. Prevents duplicate patterns from matching mismatched nodes.
4. **Commutative Permutations**:
   - For `Add` and `Mul` nodes, non-constant parent nodes are permuted via `itertools.permutations()`.
   - Limits max parents to $8$ to prevent $O(P!)$ combinatorial explosion.

### Complexity:

- **Best Case**: $O(P)$ where $P$ is pattern node depth.
- **Worst Case**: $O(N \cdot P!)$ for heavily nested commutative operators.

---

## 4. Staged Rewrite & Model Rebuilding Flow (`NeuronRewriter`)

### Flow Specifications:

1. **Edits Staging**: `replace()`, `delete()`, `insert_before()`, `register_initializer()` append edits to pending sets (`_to_remove_ids`, `_to_insert`).
2. **Topological Re-sorting**:
   - Builds dependency `DiGraph` over kept nodes + inserted nodes.
   - Executes `nx.topological_sort()`. Falls back to original insertion order if a cycle is detected.
3. **Model Reconstruction**:
   - ONNX: Calls `helper.make_graph()` and `helper.make_model()`. Executes shape inference.
   - TFLite: Serializes `OperatorCodes`, `Tensors`, `Operators`, `SubGraphs`, `Buffers`, and `Model` FlatBuffers. Copies existing option tables via FlatBuffer vtable offsets (`_copy_flatbuffer_table`).

### Complexity:

- **Build Time**: $O(V \log V + B)$ where $V$ is node count and $B$ is FlatBuffer byte length.

---

## 5. Subgraph Slicing & Boundary Tracing Flow (`trace_subgraph_boundaries`)

### Algorithmic Breakdown:

```
Inputs: nodes, tensor_map, start_points (S), end_points (E), original_outputs
Outputs: kept_ops, boundary_inputs, boundary_outputs

1. Build Bipartite DiGraph G with Op nodes (indices) and Tensor nodes (names).
2. Resolve S -> S_resolved and E -> E_resolved.
3. Compute Descendants D = {v in G | S_resolved -> v}.
4. Compute Ancestors A = {v in G | v -> E_resolved}.
5. Compute SubgraphNodes = D ∩ A.
6. Extract kept_ops = [nodes[i] for i in SubgraphNodes if type(i) == "op"].
7. Compute SubgraphTensors = {t in SubgraphNodes if type(t) == "tensor"}.
8. Identify Boundary Inputs:
     t in SubgraphTensors where producer(t) not in kept_ops AND t not in tensor_map.
9. Identify Boundary Outputs:
     t in SubgraphTensors where t in E_resolved OR producer(t) in E_resolved OR has_external_consumers(t).
10. Return (kept_ops, sorted(inputs), sorted(outputs)).
```

### Complexity:

- **Tracing Time**: $O(V + E)$ (Breadth-First / Depth-First Graph Search).
