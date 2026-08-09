# 02 — Query Engine (`NeuronQuery`)

The **Query Engine** lets you filter, traverse, sort and set-operate on nodes inside a graph using a chainable fluent API.  
Every filter returns a **new `NeuronQuery`** — nothing mutates in place.

---

## Getting a Query Object

```python
from neuron_toolkit import NeuronGraph

graph = NeuronGraph("model.onnx")
q = graph.query()          # NeuronQuery over ALL nodes
```

You can also get a query from the lower-level parser:

```python
from neuron_toolkit.parser import ONNXParser

parser = ONNXParser("model.onnx")
q = parser.find()          # same thing, lower-level
```

> Source: `test_parser.py::test_parser_find`, `test_fusion.py::test_onnx_graph_entry_point`

---

## Counting & Iterating

```python
q = graph.query()

q.count()          # int — total nodes in this query
len(q)             # same as count()

for node in q:     # iterate over nodes
    print(node.op_type, node.name)

q.all_nodes        # list of all nodes in the query
q.nodes            # same
```

---

## Filtering

### By op type

```python
q.op("Relu")                         # exact match
q.find_by_op_type("Relu")            # alias (same behaviour)

q.op("Relu").count()  # → 1
q.op("Add").count()   # → 1
```

### By name

```python
q.name("n1", exact=True)    # exact string match   → count 1
q.find_by_name("n1", exact=True)    # alias

q.name("n")                 # substring match → all nodes whose name contains "n"
q.find_by_name("n")         # alias
```

### By output rank

```python
q.rank(2)    # nodes whose first output has rank (ndim) == 2
q.rank(3)    # → 0 results if no rank-3 output exists
```

### By data type

```python
q.dtype("float32")   # nodes whose first output dtype is float32
q.dtype("int64")
```

### By tensor name (producer or consumer)

```python
# Returns nodes that PRODUCE OR CONSUME the tensor named "relu_out"
q.find_by_tensor("relu_out").count()   # e.g. 2 (Relu produces it, Add consumes it)
```

### By attribute value

```python
q.attr("kernel_shape", [3, 3])    # nodes with attribute kernel_shape == [3,3]
q.attr("kernel_shape", [5, 5])    # → 0
q.attr("non_existent")            # → 0 (attribute doesn't exist)
q.find_by_attribute("kernel_shape")  # lower-level alias
```

### By parameter presence (has initializer inputs)

```python
q.has_params()          # nodes that take at least one initializer tensor as input
q.has_params().count()  # → 1 for Add node that uses tensor B
```

### Custom predicate with `.filter()`

```python
# Any callable that receives a node and returns bool
multi_input = q.filter(lambda node: len(getattr(node, "input", [])) > 1)
multi_input.count()   # e.g. 2 (Add takes relu_out+B, Mul takes add_out+C)
```

### Combined (chained) filters

```python
# All Add nodes named "n2"
q.op("Add").name("n2").count()   # → 1

# All Add nodes with rank-2 output
q.op("Add").rank(2)
```

> Source: `test_query.py`, `test_query_deep.py`, `comprehensive_test.py::test_neuron_query_engine`

---

## Accessing Individual Nodes

```python
q.first()         # first node or None
q.single_node     # asserts exactly 1 node, returns it (raises if 0 or 2+)

# Indexing by position
q.op("Add")[0].first()   # first Add node
q.op("Add")[1].first()   # second Add node
# q[999]                 → raises IndexError
```

---

## Graph Traversal

All traversal methods return a **new `NeuronQuery`**.

### Immediate neighbours

```python
relu_q = q.name("n1", exact=True)
add_q  = q.name("n2", exact=True)

# Direct children (nodes that consume the output of this node)
relu_q.children().first().name   # "n2"

# Direct parents (nodes whose output feeds into this node)
add_q.parents().first().name     # "n1"

# Shorthand aliases used in test_query_deep.py:
relu_q.outputs()   # same as children()
add_q.inputs()     # same as parents()
```

### Multi-hop traversal

```python
# All downstream nodes reachable from n1 (Relu)
relu_q.descendants()
# → {n2 (Add), n3 (Add), n4 (Sigmoid), n5 (Tanh)}

# All upstream nodes contributing to n4 (Sigmoid)
sigmoid_q.ancestors()
# → {n0 (Identity), n1 (Relu), n2 (Add), n3 (Mul)}
```

> Source: `test_query.py::test_query_traversal`, `comprehensive_test.py::test_neuron_query_engine`

---

## Set Operations

```python
relu_q = q.op("Relu")
add_q  = q.op("Add")

# Union  |
union = relu_q | add_q
union.count()   # 2

# Intersection  &
inter = union & relu_q
inter.count()   # 1
inter.single_node.op_type  # "Relu"

# Difference  -
diff = union - relu_q
diff.count()   # 1
diff.single_node.op_type   # "Add"
```

> Source: `test_query.py::test_query_set_ops`, `comprehensive_test.py::test_neuron_query_engine`

---

## Topological Sort

```python
sorted_q = q.topological_sort()
sorted_q.count()               # same total count
sorted_q.is_topologically_sorted()   # True

# Verify un-sorted query is NOT topologically sorted
import list
rev_nodes = list(reversed(sorted_q.nodes))
rev_q = q._clone(rev_nodes)
rev_q.is_topologically_sorted()  # False
```

> Source: `test_query.py::test_query_topological_sort`

---

## Getting Parameter Tensors from a Query

```python
n1_q = q.name("n1", exact=True)    # Add node that uses tensor B

# Check if it has parameters
n1_q.has_params().count()   # 1

# Get the tensor dict: {tensor_name: np.ndarray}
params = n1_q.tensor()
print(params.keys())         # dict_keys(['B'])
print(params["B"].shape)     # (5,)

# Convenience shortcut for queries with a single node+tensor:
n1_q.single_tensor.shape     # (5,)  — the numpy array directly
```

> Source: `test_query_deep.py::test_query_params`, `test_query_single_node_tensor`

---

## Fluent Shortcuts on `NeuronGraph`

`NeuronGraph` exposes a few popular queries directly so you don't have to call `.query()` first:

```python
graph.query().op("Identity").count()   # standard

# Pattern-based where filter
q = graph.query().where(Pattern.op("Relu"))
q.count()           # 1
q.first().op_type   # "Relu"

# Pattern-based select (returns list of MatchResult)
matches = graph.query().op("Relu").select(Pattern.op("Relu"))

# Convert query to a Pattern
pat = graph.query().op("Relu").to_pattern()
pat.op_type   # "Relu"
```

> Source: `test_fusion.py::test_query_fluent_fusion`, `test_query_to_pattern`

---

## What's next?

→ [03 — Pattern DSL](./03_pattern_dsl.md) — build structural patterns to describe subgraphs.
