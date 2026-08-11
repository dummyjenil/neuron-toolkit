# Module 04: Fluent Query Engine (`NeuronQuery`)

This module covers the design, graph algorithms, and method-chaining filtering capabilities of `NeuronQuery` (`src/neuron_toolkit/query/core.py`).

---

## 1. What is `NeuronQuery`?

`NeuronQuery` is an immutable, chainable query builder inspired by ORMs (like Django ORM or SQLAlchemy) and jQuery. It allows engineers to query neural network graphs using simple method calls.

```python
# Chainable filter example:
conv_layers = (
    graph.query()
    .op("Conv")
    .attr("group", 1)
    .rank(4)
    .dtype("float32")
)
```

---

## 2. Graph Construction & Caching Mechanics

When a `NeuronQuery` is created, it initializes three cached graph representations using `@cached_property`:

### 1. `output_map`: Fast Producer Lookup
Maps every output tensor name in the graph to the operator that produced it:
```python
output_map = {out: node for node in all_nodes for out in node.output}
```
**Time Complexity**: $O(N)$ build, $O(1)$ lookup.

### 2. `_nx_graph`: NetworkX Directed Graph
Builds a NetworkX `DiGraph` representation where nodes represent operators and directed edges represent tensor dataflow:
```python
g = nx.DiGraph()
for node in all_nodes:
    g.add_node(node.name, proto=node)
    for inp in node.input:
        if parent := output_map.get(inp):
            g.add_edge(parent.name, node.name, tensor=inp)
```

### 3. `_node_to_idx`: Global Topological Index Map
Ranks all nodes by their topological execution order:
```python
order = list(nx.topological_sort(_nx_graph))
_node_to_idx = {name: i for i, name in enumerate(order)}
```

### Immutable Query Cloning (`_clone`)
Whenever a filter is applied, `NeuronQuery` returns a new `NeuronQuery` instance containing only the matching subset of nodes, while **sharing expensive cached properties** (`output_map`, `_nx_graph`, `_node_to_idx`) to ensure zero performance overhead:

```python
def _clone(self, nodes: Sequence[object]) -> NeuronQuery:
    new = self.__class__(nodes, self.tensor_map, self.all_nodes, ...)
    for attr in ("output_map", "_nx_graph", "_node_to_idx"):
        if attr in self.__dict__:
            new.__dict__[attr] = self.__dict__[attr]
    return new
```

---

## 3. Filtering Capabilities

### Operator Type & Name Filtering
- `.op("Conv")`: Filters nodes where `op_type == "Conv"`.
- `.name("conv1", exact=False)`: Case-insensitive substring search (or exact match).

### Attribute & Parameter Filtering
- `.attr("group", 1)`: Filters nodes having attribute `group == 1`. Supports callable predicates: `.attr("kernel_shape", lambda k: k[0] > 3)`.
- `.find_by_param_name("weight")`: Finds operators connected to a initializer parameter matching `"weight"`.
- `.has_params()`: Filters operators that possess constant weight tensors.

### Tensor Shape & DataType Filtering
- `.rank(4)`: Filters operators whose primary output tensor has rank 4.
- `.dtype("float32")`: Filters operators producing `"float32"` outputs.

---

## 4. Graph Traversal API

`NeuronQuery` allows navigating relative relationships in the computation DAG:

```
[Parent Node] ---> [Current Node] ---> [Child Node]
```

- `.parents()` / `.inputs()`: Direct predecessors feeding into the selected nodes.
- `.children()` / `.outputs()`: Direct successors receiving outputs from the selected nodes.
- `.ancestors(max_depth=100)`: Multi-source Breadth-First Search (BFS) gathering all upstream nodes.
- `.descendants(max_depth=100)`: Multi-source BFS gathering all downstream nodes.

### Entry & Exit Nodes:
- `.entry_nodes`: Nodes consuming global model inputs.
- `.output_nodes`: Nodes producing global model outputs.

---

## 5. Set Operations & Overloaded Operators

You can combine or compare queries using set algebra:

```python
q1 = graph.query().op("Conv")
q2 = graph.query().op("MatMul")

# Set operations:
q_union = q1 | q2          # Or q1.union(q2)
q_inter = q1 & q2          # Or q1.intersection(q2)
q_diff  = q1 - q2          # Or q1.difference(q2)
```

---

## 6. Topological Sorting Verification

You can verify and sort nodes in topological dependency order:
```python
# Check if query nodes are topologically sorted:
if not query.is_topologically_sorted():
    query = query.topological_sort()
```
The sorting uses `_node_to_idx` to order selected nodes in $O(K \log K)$ time.
