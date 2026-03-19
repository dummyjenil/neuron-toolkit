# onnx_toolkit

A unified ONNX parsing and pattern-matching library. `onnx_toolkit` provides two tightly integrated components:

- **`ONNXParser` / `ONNXQuery`** — load a model and query its graph with a fluent, chainable API
- **`Pattern` / `Pattern.detect`** — describe structural subgraph patterns and match them against the graph using DFS with commutativity support

## Installation

```bash
pip install onnx_toolkit
```

## Quick Start

```python
from onnx_toolkit import ONNXParser, Pattern

# Load a model and query it
parser = ONNXParser("model.onnx")

# Find all Conv nodes that have weights, then look at their immediate children
convs = parser.find().find_by_op_type("Conv").has_params()
print(convs)

# Check if a node matches a Swish/SiLU activation pattern
x = Pattern.any()
swish = Pattern.silu(x)
detector = Pattern.detect(parser.model, start_node="MatMul_0")
print(detector.match(swish))  # True or False
```

---

## Logging

`onnx_toolkit` uses Python's standard `logging` module. To enable debug output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

To enable output only for this library:

```python
logging.getLogger("onnx_toolkit").setLevel(logging.DEBUG)
logging.getLogger("onnx_toolkit").addHandler(logging.StreamHandler())
```

### Sub-loggers

| Logger name             | Covers                                   |
|-------------------------|------------------------------------------|
| `onnx_toolkit`          | Top-level / parser messages              |
| `onnx_toolkit.query`    | `ONNXQuery` filter and traversal steps   |
| `onnx_toolkit.pattern`  | `Pattern` construction messages          |
| `onnx_toolkit.detect`   | `Pattern.detect` DFS walk and decisions  |

---

## `ONNXParser`

Loads an ONNX model from disk and provides an entry point for graph queries.

```python
parser = ONNXParser("model.onnx")
```

### Methods

#### `find() → ONNXQuery`

Returns an `ONNXQuery` over **all** nodes in the graph. This is the starting point for every query chain.

```python
query = parser.find()
```

#### `pattern_detect(pattern, start_node=None, end_node=None) → bool`

Convenience shortcut: creates a `Pattern.detect` bound to this model and immediately calls `.match(pattern)`.

```python
x = Pattern.any()
result = parser.pattern_detect(Pattern.relu(x), start_node="Relu_0")
```

| Parameter    | Type                      | Description                              |
|--------------|---------------------------|------------------------------------------|
| `pattern`    | `Pattern`                 | The pattern to match                     |
| `start_node` | `str` or `NodeProto`      | Seed node (name string or proto object)  |
| `end_node`   | `str` or `NodeProto`      | Stop traversal at this node (optional)   |

#### `summary() → str`

Returns a human-readable text summary of the loaded model, including node count, tensor count, graph I/O counts, and a ranked frequency table of op types.

```python
print(parser.summary())
# ONNX model summary
#   Nodes       : 152
#   Tensors     : 104
#   ...
```

---

## `ONNXQuery`

A lazy, chainable view over a subset of ONNX graph nodes. Every filter and traversal method returns a **new** `ONNXQuery`, so calls can be freely chained without mutating the receiver.

```python
parser.find()
      .find_by_op_type("Conv")
      .has_params()
      .children()
      .find_by_op_type("Relu")
```

### Filter Methods

#### `find_by_op_type(op_type: str) → ONNXQuery`

Filters nodes whose `op_type` matches exactly (case-sensitive).

```python
convs = parser.find().find_by_op_type("Conv")
```

#### `find_by_name(name: str, *, exact: bool = False) → ONNXQuery`

Filters nodes by name. By default performs a case-insensitive substring match. Pass `exact=True` for strict equality.

```python
# Substring match (default)
nodes = parser.find().find_by_name("block2")

# Exact match
node = parser.find().find_by_name("Conv_14", exact=True)
```

#### `find_by_tensor(tensor_name: str) → ONNXQuery`

Filters nodes that either consume or produce a tensor with the given name.

```python
nodes = parser.find().find_by_tensor("input.1")
```

#### `find_by_param_name(name: str, *, exact: bool = False) → ONNXQuery`

Filters nodes whose weight (initializer) inputs match the given name. Supports substring (default) or exact matching.

```python
# All nodes with a weight whose name contains "bias"
nodes = parser.find().find_by_param_name("bias")
```

#### `find_by_attribute(attr_name: str, value: Any = None) → ONNXQuery`

Filters nodes that have the given ONNX attribute, optionally requiring it to equal a specific value.

```python
# Any node with a "group" attribute
parser.find().find_by_attribute("group")

# Nodes with group=2 (depthwise conv)
parser.find().find_by_attribute("group", value=2)
```

#### `has_params() → ONNXQuery`

Filters to nodes that have at least one weight tensor (initializer input).

```python
weighted = parser.find().find_by_op_type("Conv").has_params()
```

### Traversal Methods

#### `children() → ONNXQuery`

Returns all nodes that consume any output produced by the current selection (one hop forward).

```python
next_layer = query.children()
```

#### `parents() → ONNXQuery`

Returns all nodes that produce any input consumed by the current selection (one hop backward).

```python
prev_layer = query.parents()
```

#### `ancestors(max_depth: int = 100) → ONNXQuery`

Returns all transitive parent nodes up to `max_depth` hops away. The current nodes are **not** included.

```python
all_predecessors = query.ancestors()
close_predecessors = query.ancestors(max_depth=3)
```

#### `descendants(max_depth: int = 100) → ONNXQuery`

Returns all transitive child nodes up to `max_depth` hops away. The current nodes are **not** included.

```python
all_successors = query.descendants()
```

### Entry / Exit Shortcuts

#### `entry_nodes → ONNXQuery` *(property)*

Returns nodes whose inputs include at least one graph-level input.

```python
first_ops = parser.find().entry_nodes
```

#### `output_nodes → ONNXQuery` *(property)*

Returns nodes whose outputs include at least one graph-level output.

```python
last_ops = parser.find().output_nodes
```

### Tensor / Parameter Access

#### `tensor(name: str = None) → Any`

Returns weight tensor(s) for the selected nodes as NumPy arrays.

- If `name` is provided: returns that specific tensor (or `None` if not found).
- If exactly **one** node is selected: returns `{param_name: array, ...}`.
- If **multiple** nodes are selected: returns `{node_name: {param_name: array}, ...}`.

```python
# Get a tensor by name directly
w = query.tensor("conv1.weight")

# Get all weights for a single selected node
params = parser.find().find_by_name("Conv_0", exact=True).tensor()
```

#### `single_tensor → np.ndarray` *(property)*

Returns the single weight tensor for a single selected node. Raises `ValueError` if the selection contains more or fewer than one node, or if the node has multiple or no tensors.

```python
weight = parser.find().find_by_name("Conv_0", exact=True).single_tensor
```

### Pattern Matching Integration

#### `matches(pattern: Pattern) → ONNXQuery`

Returns the subset of nodes from the current selection whose subgraphs match `pattern`. This is the preferred way to do bulk pattern matching across many nodes.

```python
x = Pattern.any()
silu_nodes = parser.find().find_by_op_type("Mul").matches(Pattern.silu(x))
```

### Set Operations

#### `union(other: ONNXQuery) → ONNXQuery`

Returns all nodes present in `self` **or** `other`, deduplicated.

#### `intersection(other: ONNXQuery) → ONNXQuery`

Returns nodes present in both `self` **and** `other`.

#### `difference(other: ONNXQuery) → ONNXQuery`

Returns nodes present in `self` but **not** in `other`.

```python
a = parser.find().find_by_op_type("Relu")
b = parser.find().find_by_op_type("Conv").children().find_by_op_type("Relu")

only_in_a = a.difference(b)
all_relus = a.union(b)
shared = a.intersection(b)
```

### Accessors and Helpers

| Method / Property     | Returns              | Description                                             |
|-----------------------|----------------------|---------------------------------------------------------|
| `count()`             | `int`                | Number of nodes in the current selection                |
| `is_empty()`          | `bool`               | True if the selection is empty                          |
| `op_types()`          | `list[str]`          | Deduplicated list of op types in the selection          |
| `first()`             | `NodeProto` or None  | First node, or `None` if empty                          |
| `last()`              | `NodeProto` or None  | Last node, or `None` if empty                           |
| `single_node`         | `NodeProto`          | The single selected node; raises if count ≠ 1           |
| `__iter__`            | iterator             | Iterate over selected `NodeProto` objects               |
| `__len__`             | `int`                | Same as `count()`                                       |
| `__getitem__(i)`      | `ONNXQuery`          | Slice or index the selection                            |

---

## `Pattern`

A lightweight DSL for describing ONNX subgraph structures. Patterns are composable and can be built using arithmetic operators, class methods, and helper constructors.

### Named Constructors

#### `Pattern.any() → Pattern`

Wildcard that matches any node regardless of op type.

```python
x = Pattern.any()
```

#### `Pattern.const(value) → Pattern`

Matches a constant or initializer tensor whose value is approximately equal to `value` (tolerance `1e-3`).

```python
half = Pattern.const(0.5)
```

#### `Pattern.op(op_type, *input_patterns) → Pattern`

Matches a node with the given `op_type`, optionally constraining its parent nodes via `input_patterns`.

```python
erf_of_x = Pattern.op("Erf", Pattern.any())
```

Patterns created with `Pattern.op` can also be used as unary functions by calling them:

```python
relu = Pattern.op("Relu")
applied = relu(Pattern.any())   # same as Pattern.op("Relu", Pattern.any())
```

> **Note:** A pattern can only be called as a unary function if it has no existing inputs. Use `Pattern.op(op_type, *inputs)` directly for multi-input patterns.

### Operator Overloads (DSL Sugar)

You can compose patterns using standard Python operators:

| Expression      | Resulting Pattern |
|-----------------|-------------------|
| `a + b`         | `Add(a, b)`       |
| `a * b`         | `Mul(a, b)`       |
| `a ** b`        | `Pow(a, b)`       |
| `a - b`         | `Sub(a, b)`       |
| `a / b`         | `Div(a, b)`       |
| `-a`            | `Neg(a)`          |

Raw Python scalars are automatically coerced to `Pattern.const(value)`:

```python
x = Pattern.any()
cube = x ** 3.0    # equivalent to x ** Pattern.const(3.0)
```

### Built-in Activation Patterns

Commonly used activation functions are available as class methods. All accept a `Pattern` argument representing the input.

| Method                         | Activation                |
|--------------------------------|---------------------------|
| `Pattern.relu(x)`              | ReLU                      |
| `Pattern.relu6(x)`             | ReLU6 (Clip 0–6)          |
| `Pattern.sigmoid(x)`           | Sigmoid                   |
| `Pattern.tanh(x)`              | Tanh                      |
| `Pattern.leaky_relu(x)`        | LeakyReLU                 |
| `Pattern.elu(x)`               | ELU                       |
| `Pattern.selu(x)`              | SELU                      |
| `Pattern.softplus(x)`          | Softplus                  |
| `Pattern.softsign(x)`          | Softsign                  |
| `Pattern.hardsigmoid(x)`       | HardSigmoid               |
| `Pattern.hardswish(x)`         | HardSwish (`x * σ_hard(x)`) |
| `Pattern.silu(x)`              | SiLU (`x * sigmoid(x)`)  |
| `Pattern.swish(x)`             | Swish (alias of SiLU)     |
| `Pattern.gelu(x)`              | GELU (erf approximation)  |
| `Pattern.gelu_tanh(x)`         | GELU (tanh approximation) |
| `Pattern.mish(x)`              | Mish (`x * tanh(softplus(x))`) |
| `Pattern.softmax(x)`           | Softmax                   |
| `Pattern.log_softmax(x)`       | LogSoftmax                |
| `Pattern.prelu(x, slope)`      | PReLU                     |
| `Pattern.thresholded_relu(x)`  | ThresholdedReLU           |

### Example: Building a Custom Pattern

```python
x = Pattern.any()

# Manual GELU (erf form)
gelu = x * (Pattern.op("Erf")(x / Pattern.const(1.41421356237)) + Pattern.const(1.0)) * Pattern.const(0.5)

# Or use the built-in helper
gelu = Pattern.gelu(x)
```

---

## `Pattern.detect`

Matches a `Pattern` against a specific subgraph rooted at a given node, using DFS with commutativity support for `Add` and `Mul` nodes.

```python
detector = Pattern.detect(model, start_node="MatMul_0")
result = detector.match(pattern)
```

### Constructor

```python
Pattern.detect(model, start_node=None, end_node=None)
```

| Parameter    | Type                         | Description                                                     |
|--------------|------------------------------|-----------------------------------------------------------------|
| `model`      | `ModelProto` or `_GraphShim` | The ONNX model to search in                                     |
| `start_node` | `str` or `NodeProto`         | Root of the subgraph to check (name string or proto object)     |
| `end_node`   | `str` or `NodeProto`         | Optional stop condition — branch succeeds when this node is reached |

### `match(pattern: Pattern) → bool`

Returns `True` if the subgraph rooted at `start_node` matches `pattern`.

```python
x = Pattern.any()
detector = Pattern.detect(model, start_node="Mul_5")
print(detector.match(Pattern.silu(x)))  # True or False
```

### `find_all(pattern: Pattern) → list[NodeProto]`

Scans nodes for all subgraph matches against `pattern`. If `start_node` was provided, only descendants of that node are scanned; otherwise the entire graph is searched.

```python
detector = Pattern.detect(model)
all_gelus = detector.find_all(Pattern.gelu(Pattern.any()))
```

---

## Recipes

### Find all depthwise convolutions

```python
parser = ONNXParser("model.onnx")
dw_convs = parser.find().find_by_op_type("Conv").find_by_attribute("group", value=-1)
# Note: group=in_channels for depthwise; use find_by_attribute("group") to list all grouped convs
```

### Retrieve weights of a specific layer

```python
conv_node = parser.find().find_by_name("features.0.0", exact=True)
params = conv_node.tensor()
# {"features.0.0.weight": array(...), "features.0.0.bias": array(...)}
```

### Find all SiLU / Swish activations in the graph

```python
x = Pattern.any()
swish_muls = (
    parser.find()
    .find_by_op_type("Mul")
    .matches(Pattern.silu(x))
)
print(f"Found {swish_muls.count()} SiLU/Swish activation(s)")
```

### Walk from graph inputs to the first Conv

```python
entry = parser.find().entry_nodes
first_conv = entry.descendants().find_by_op_type("Conv").first()
```

### Check whether any GELU variants exist

```python
x = Pattern.any()
detector = Pattern.detect(parser.model)

has_gelu     = bool(detector.find_all(Pattern.gelu(x)))
has_gelu_approx = bool(detector.find_all(Pattern.gelu_tanh(x)))
```

---

## API Reference Summary

### `ONNXParser`

| Method                                     | Returns        |
|--------------------------------------------|----------------|
| `ONNXParser(onnx_path)`                    | instance       |
| `find()`                                   | `ONNXQuery`    |
| `pattern_detect(pattern, start_node, end_node)` | `bool`    |
| `summary()`                                | `str`          |

### `ONNXQuery`

| Method / Property                          | Returns        |
|--------------------------------------------|----------------|
| `find_by_op_type(op_type)`                 | `ONNXQuery`    |
| `find_by_name(name, exact=False)`          | `ONNXQuery`    |
| `find_by_tensor(tensor_name)`              | `ONNXQuery`    |
| `find_by_param_name(name, exact=False)`    | `ONNXQuery`    |
| `find_by_attribute(attr_name, value=None)` | `ONNXQuery`    |
| `has_params()`                             | `ONNXQuery`    |
| `children()`                               | `ONNXQuery`    |
| `parents()`                                | `ONNXQuery`    |
| `ancestors(max_depth=100)`                 | `ONNXQuery`    |
| `descendants(max_depth=100)`               | `ONNXQuery`    |
| `entry_nodes`                              | `ONNXQuery`    |
| `output_nodes`                             | `ONNXQuery`    |
| `tensor(name=None)`                        | `dict` / array |
| `single_tensor`                            | `np.ndarray`   |
| `matches(pattern)`                         | `ONNXQuery`    |
| `union(other)`                             | `ONNXQuery`    |
| `intersection(other)`                      | `ONNXQuery`    |
| `difference(other)`                        | `ONNXQuery`    |
| `count()`                                  | `int`          |
| `is_empty()`                               | `bool`         |
| `op_types()`                               | `list[str]`    |
| `first()`                                  | `NodeProto`    |
| `last()`                                   | `NodeProto`    |
| `single_node`                              | `NodeProto`    |

### `Pattern`

| Constructor / Method                        | Description                                     |
|---------------------------------------------|-------------------------------------------------|
| `Pattern.any()`                             | Wildcard                                        |
| `Pattern.const(value)`                      | Constant tensor matcher                         |
| `Pattern.op(op_type, *inputs)`              | Op-type matcher with optional input constraints |
| Arithmetic operators (`+`, `*`, `**`, etc.) | Compose Add / Mul / Pow / Sub / Div / Neg nodes |
| Activation helpers (see table above)        | Pre-built common activation patterns            |

### `Pattern.detect`

| Method                    | Returns           |
|---------------------------|-------------------|
| `match(pattern)`          | `bool`            |
| `find_all(pattern)`       | `list[NodeProto]` |