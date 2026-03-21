# ONNXQuery

`ONNXQuery` is a lazy, chainable view over a subset of ONNX graph nodes. Every method returns a new `ONNXQuery` without mutating the receiver, so calls compose freely in a pipeline style.

You never construct an `ONNXQuery` directly. Always start from `ONNXParser.find()`.

---

## Chaining Model

Each method returns a new `ONNXQuery` scoped to a subset of nodes:

```python
parser.find()
      .find_by_op_type("Conv")
      .has_params()
      .children()
      .find_by_op_type("BatchNormalization")
      .apply(lambda n, _: print(n.name))
```

The underlying graph data (all nodes, tensor map, shape info) is shared across every query in a chain without copying.

---

## Filters

### `find_by_op_type(op_type: str) → ONNXQuery`

Keep nodes whose `op_type` equals `op_type` exactly (case-sensitive).

```python
parser.find().find_by_op_type("Conv")
parser.find().find_by_op_type("BatchNormalization")
```

---

### `find_by_name(name: str, *, exact: bool = False) → ONNXQuery`

Keep nodes whose name matches `name`.

- `exact=False` (default): case-insensitive substring search.
- `exact=True`: exact equality.

```python
parser.find().find_by_name("layer1")          # all nodes with "layer1" in name
parser.find().find_by_name("/layer1/conv1", exact=True)
```

---

### `find_by_tensor(tensor_name: str) → ONNXQuery`

Keep nodes that consume **or** produce a tensor named `tensor_name`.

```python
parser.find().find_by_tensor("input")
parser.find().find_by_tensor("conv1_weight")
```

---

### `find_by_param_name(name: str, *, exact: bool = False) → ONNXQuery`

Keep nodes whose initializer (weight) inputs match `name`.

- `exact=False` (default): case-insensitive substring.
- `exact=True`: exact match.

```python
parser.find().find_by_param_name("bias")
parser.find().find_by_param_name("layer1.0.weight", exact=True)
```

---

### `find_by_attribute(attr_name: str, value=None) → ONNXQuery`

Keep nodes that have attribute `attr_name`. Optionally filter by value.

- `value=None`: presence check only — any node with this attribute passes.
- `value=<literal>`: attribute value must equal the literal.
- `value=<callable>`: attribute value is passed to the callable; node passes if the callable returns `True`.

```python
# All nodes with a "group" attribute
parser.find().find_by_attribute("group")

# Nodes where group == 1 (standard convolutions)
parser.find().find_by_attribute("group", 1)

# Nodes where group > 1 (depthwise / grouped convolutions)
parser.find().find_by_attribute("group", lambda g: g > 1)

# Nodes with dilated convolutions
parser.find().find_by_attribute("dilations", lambda d: any(v > 1 for v in d))
```

---

### `has_params() → ONNXQuery`

Keep nodes that have at least one weight tensor (initializer input).

```python
parser.find().find_by_op_type("Conv").has_params()
parser.find().find_by_op_type("Gemm").has_params()
```

---

## Traversal

All traversal methods operate on the **full graph** (`all_nodes`), not just the currently selected subset, so they can discover nodes that were not part of the prior selection.

### `children() → ONNXQuery`

Nodes in the graph that consume any output produced by the currently selected nodes.

```python
# Direct consumers of every Conv
parser.find().find_by_op_type("Conv").children()
```

---

### `parents() → ONNXQuery`

Nodes in the graph that produce any input consumed by the currently selected nodes.

```python
# What feeds into every BatchNorm?
parser.find().find_by_op_type("BatchNormalization").parents()
```

---

### `ancestors(max_depth: int = 100) → ONNXQuery`

All transitive upstream nodes up to `max_depth` hops away. The current nodes are **not** included in the result.

```python
# Everything upstream of a specific output node, at most 5 hops
output_node = parser.find().output_nodes
output_node.ancestors(max_depth=5)
```

---

### `descendants(max_depth: int = 100) → ONNXQuery`

All transitive downstream nodes up to `max_depth` hops away. The current nodes are **not** included.

```python
# Everything reachable from the first Conv
first_conv = parser.find().find_by_op_type("Conv")[0]
first_conv.descendants(max_depth=3)
```

---

### `entry_nodes` *(property)* `→ ONNXQuery`

Nodes whose inputs include at least one graph-level input tensor. Always scans all nodes regardless of the current selection.

```python
parser.find().entry_nodes
```

---

### `output_nodes` *(property)* `→ ONNXQuery`

Nodes whose outputs include at least one graph-level output tensor. Always scans all nodes.

```python
parser.find().output_nodes
```

---

## Weight Tensor Access

### `tensor(name: str | None = None) → dict | np.ndarray | None`

Return weight tensors for the selected nodes.

| Call form | Return value |
|---|---|
| `tensor("conv1.weight")` | Single `np.ndarray` or `None` |
| `tensor()` on 1 node | `{param_name: np.ndarray, …}` |
| `tensor()` on N nodes | `{node_name: {param_name: np.ndarray}, …}` |

```python
# Direct tensor lookup by name
w = parser.find().tensor("conv1.weight")
print(w.shape)   # (64, 3, 7, 7)

# All weights of a single Conv node
params = parser.find().find_by_name("conv1", exact=True).tensor()
print(params.keys())  # dict_keys(["weight", "bias"])

# Weights of all Conv nodes
all_params = parser.find().find_by_op_type("Conv").tensor()
# {"conv1": {"weight": …, "bias": …}, "conv2": …}
```

---

### `single_tensor` *(property)* `→ np.ndarray`

Asserts that exactly one node is selected and it has exactly one weight tensor, then returns that tensor. Raises `ValueError` otherwise.

```python
w = parser.find().find_by_name("conv1", exact=True).single_tensor
```

---

## Pattern Matching

### `matches(pattern: Pattern) → ONNXQuery`

Return the subset of currently selected nodes whose local subgraph matches `pattern`. Returns a new `ONNXQuery` of matching start nodes only.

The output→node map is built once and shared across all candidate tests.

```python
x   = Pattern.any().capture("x")
hit = parser.find().matches(Pattern.relu(x))
print(hit.count(), "Relu nodes matched")
```

---

### `match_results(pattern: Pattern) → list[MatchResult]`

Like `matches()` but returns a `list[MatchResult]` giving full access to the matched subgraph, named captures, and boundary nodes.

The `_tensor_map` is attached to every result so that `result.as_query()` includes weight tensors.

```python
x       = Pattern.any().capture("x")
results = parser.find().match_results(Pattern.gelu(x))

for r in results:
    print(r.start.name)
    print("  input:", r.bindings["x"].name)
    print("  subgraph nodes:", [n.name for n in r.nodes])
    sub_query = r.as_query()
    print("  op types:", sub_query.op_types())
```

---

## Ordering

### `topological_sort() → ONNXQuery`

Return a new `ONNXQuery` with the selected nodes re-ordered into topological order using Kahn's algorithm. Only edges within the current selection are considered.

Raises `ValueError` if a cycle is detected (which cannot occur in valid ONNX graphs).

```python
sorted_nodes = parser.find().topological_sort()
```

---

### `is_topologically_sorted() → bool`

Return `True` if the selected nodes are already in topological order.

```python
if not parser.find().is_topologically_sorted():
    print("Graph nodes are not in topological order")
```

---

## Functional Transform

### `apply(fn: Callable[[NodeProto, TensorMap], Any]) → ONNXQuery`

Call `fn(node, params)` on every selected node. Returns `self` unchanged so the call can be inserted into a chain without breaking it.

`params` is a `dict[str, np.ndarray]` of the node's initializer inputs (empty if the node has none).

```python
# Print all Conv weight shapes
parser.find().find_by_op_type("Conv").apply(
    lambda n, p: print(n.name, p.get("weight", {}).shape if "weight" in p else "—")
)

# Collect node names into a list
names = []
parser.find().has_params().apply(lambda n, _: names.append(n.name))
```

---

## Set Operations

All set operations accept another `ONNXQuery` and deduplicate by node name.

### `union(other: ONNXQuery) → ONNXQuery`

Nodes present in `self` **or** `other`.

### `intersection(other: ONNXQuery) → ONNXQuery`

Nodes present in `self` **and** `other`.

### `difference(other: ONNXQuery) → ONNXQuery`

Nodes present in `self` but **not** in `other`.

```python
convs  = parser.find().find_by_op_type("Conv")
with_b = parser.find().find_by_param_name("bias")

# Convolutions without a bias initializer
no_bias = convs.difference(with_b)

# Nodes that are either Conv or Gemm
linear = convs.union(parser.find().find_by_op_type("Gemm"))
```

---

## Export

### `to_dot(*, show_params: bool = False) → str`

Return a Graphviz DOT string visualising the selected nodes. Edges represent data flow between nodes in the selection.

- `show_params=True`: also draw initializer tensors as ellipse-shaped leaf nodes.

```python
dot = parser.find().to_dot(show_params=True)
# Paste into https://dreampuf.github.io/GraphvizOnline/

# Or write to a file and render with graphviz CLI
with open("graph.dot", "w") as f:
    f.write(dot)
# dot -Tpng graph.dot -o graph.png
```

---

### `to_dataframe() → pandas.DataFrame`

Return a pandas DataFrame with one row per selected node. Requires `pandas` to be installed.

Columns:

| Column | Type | Description |
|---|---|---|
| `name` | `str` | Node name |
| `op_type` | `str` | ONNX op type |
| `num_inputs` | `int` | Number of input edges (including initializers) |
| `num_outputs` | `int` | Number of output edges |
| `num_params` | `int` | Number of initializer (weight) inputs |
| `param_shapes` | `dict` | `{name: [dim, …]}` for each weight |
| `attributes` | `dict` | All ONNX attributes as Python values |

```python
df = parser.find().find_by_op_type("Conv").to_dataframe()
print(df[["name", "num_params", "param_shapes"]])
```

---

### `to_onnx(path: str) → None`

Save the selected subgraph as a minimal standalone `.onnx` file.

Subgraph boundaries are inferred automatically:
- A tensor is a **subgraph input** if it is consumed within the selection but not produced within it, and is not an initializer.
- A tensor is a **subgraph output** if it is produced within the selection but not consumed within it.

All required initializers are included in the output file.

```python
# Export only the first few layers
early = parser.find().entry_nodes.descendants(max_depth=4)
early.union(parser.find().entry_nodes).to_onnx("early_layers.onnx")
```

> **Note:** All exported tensor type annotations use `FLOAT`. If your subgraph uses other dtypes, load the exported file and update the value_info annotations as needed.

---

## Accessors and Iteration

### `single_node` *(property)* `→ NodeProto`

Assert exactly one node is selected and return it. Raises `ValueError` otherwise.

### `first() → NodeProto | None`

Return the first node, or `None` if the query is empty.

### `last() → NodeProto | None`

Return the last node, or `None` if the query is empty.

### `count() → int`

Number of nodes in the current selection. Equivalent to `len(query)`.

### `is_empty() → bool`

True if the selection contains no nodes.

### `op_types() → list[str]`

Deduplicated list of op type strings in the order they first appear in the selection.

```python
parser.find().op_types()
# ["Conv", "BatchNormalization", "Relu", "Add", "GlobalAveragePool", "Flatten", "Gemm"]
```

### Iteration

`ONNXQuery` is iterable and supports `len()`, `bool()`, and indexing:

```python
for node in parser.find().find_by_op_type("Relu"):
    print(node.name)

first_conv  = parser.find().find_by_op_type("Conv")[0]    # ONNXQuery of 1 node
first_three = parser.find()[0:3]                           # ONNXQuery of 3 nodes

if parser.find().find_by_op_type("LSTM"):
    print("Model contains LSTM nodes")
```
