# `NeuronQuery` - Fluent Query Engine

`NeuronQuery` provides a chainable API for inspecting, filtering, traversing, and selecting nodes in neural network graphs.

---

## Obtaining a Query

You can obtain a `NeuronQuery` instance from a `NeuronGraph` or backend parser:

```python
from neuron_toolkit import NeuronGraph

graph = NeuronGraph.load("model.onnx")
q = graph.query()
```

---

## Filtering API

All filter methods return a new `NeuronQuery` instance containing the filtered subset of nodes.

| Method               | Syntax                                                 | Description                                                                   |
| :------------------- | :----------------------------------------------------- | :---------------------------------------------------------------------------- |
| **Op Type**          | `.op(op_type)` / `.find_by_op_type(op_type)`           | Filter nodes by operator type (e.g. `"Conv"`, `"Add"`).                       |
| **Name**             | `.name(name, exact=False)` / `.find_by_name(name)`     | Filter nodes by name (substring match by default, exact if specified).        |
| **Attribute**        | `.attr(name, value=None)` / `.find_by_attribute(name)` | Filter nodes possessing an attribute, optionally matching value or predicate. |
| **Rank**             | `.rank(rank)`                                          | Filter nodes whose output tensor has a specific rank (e.g. `4`).              |
| **Data Type**        | `.dtype(dtype)`                                        | Filter nodes by output tensor data type (e.g. `"float32"`, `"int8"`).         |
| **Tensor**           | `.find_by_tensor(name)`                                | Filter nodes consuming or producing tensor `name`.                            |
| **Param Name**       | `.find_by_param_name(name, exact=False)`               | Filter nodes bound to weight tensor `name`.                                   |
| **Params Check**     | `.has_params()`                                        | Filter nodes that possess parameter/weight tensors.                           |
| **Custom Predicate** | `.filter(predicate_fn)`                                | Filter nodes using a custom Python callable `(node) -> bool`.                 |

### Filtering Examples

```python
# Conv nodes with output rank 4 and group attribute = 1
q_conv = graph.query().op("Conv").rank(4).attr("group", 1)

# Nodes producing float32 tensors with 'weight' in their parameter names
q_weights = graph.query().dtype("float32").find_by_param_name("weight")
```

---

## Graph Traversal API

`NeuronQuery` supports graph traversal through NetworkX-backed reachability routines.

- **`.children()` / `.outputs()`**: Returns direct successor (downstream) nodes.
- **`.parents()` / `.inputs()`**: Returns direct predecessor (upstream) nodes.
- **`.descendants(max_depth=100)`**: Returns all downstream nodes reachable within `max_depth`.
- **`.ancestors(max_depth=100)`**: Returns all upstream nodes reachable within `max_depth`.
- **`.entry_nodes`**: Returns nodes that consume graph inputs.
- **`.output_nodes`**: Returns nodes that produce final graph outputs.

### Traversal Example

```python
# Find all descendants of graph entry nodes
all_downstream = graph.query().entry_nodes.descendants()

# Find immediate predecessors of output nodes
final_parents = graph.query().output_nodes.parents()
```

---

## Set Operations & Operator Overloads

`NeuronQuery` supports standard set operations for combining or narrowing node selections.

- **Union (`query1 | query2` or `query1.union(query2)`)**: Set of nodes present in either query.
- **Intersection (`query1 & query2` or `query1.intersection(query2)`)**: Set of nodes present in both queries.
- **Difference (`query1 - query2` or `query1.difference(query2)`)**: Set of nodes in `query1` but not in `query2`.

```python
add_nodes = graph.query().op("Add")
mul_nodes = graph.query().op("Mul")

# Combine Add and Mul nodes
math_nodes = add_nodes | mul_nodes
```

---

## Topological Analysis & Ordering

- **`.topological_sort() -> NeuronQuery`**: Returns query nodes sorted in global topological execution order.
- **`.is_topologically_sorted() -> bool`**: Returns `True` if current node selection is topologically ordered.

```python
sorted_query = graph.query().op("Conv").topological_sort()
```

---

## Accessors & Utilities

- **`.count() -> int`**: Returns number of selected nodes.
- **`.is_empty() -> bool`**: Returns `True` if selection contains 0 nodes.
- **`.op_types() -> list[str]`**: Unique op types present in query selection.
- **`.first() -> object | None`**: First node in selection.
- **`.last() -> object | None`**: Last node in selection.
- **`.single_node -> object`**: Asserts exactly 1 node in query and returns it.
- **`.single_tensor -> np.ndarray`**: Asserts exactly 1 weight tensor in query node and returns NumPy array.
- **`.apply(fn: Callable[[node, params], None])`**: Executes callback function on each node and its parameter tensors.
- **`.to_pattern() -> Pattern`**: Converts the first node in selection into a symbolic `Pattern`.
