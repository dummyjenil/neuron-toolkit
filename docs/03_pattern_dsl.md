# 03 — Pattern DSL

The **Pattern DSL** is how you describe the subgraph you want to find or replace.  
A `Pattern` is a tree of `Pattern` nodes — each represents one operator in the graph.

---

## Import

```python
from neuron_toolkit import Pattern
# or
from neuron_toolkit.pattern import Pattern
```

---

## Basic Pattern Constructors

### `Pattern.op(op_type, *input_patterns)`

Match a node with the given op type. Optionally specify what its inputs should look like.

```python
# Match any Relu node (no constraint on its inputs)
p = Pattern.op("Relu")

# Match Add(anything, anything)
p = Pattern.op("Add", Pattern.any(), Pattern.any())

# Match Mul(Add(...), const=2.0)
p = Pattern.op("Mul",
    Pattern.op("Add"),
    Pattern.const(2.0)
)
```

### `Pattern.any()`

A **wildcard** — matches any single node (of any op type).

```python
p = Pattern.any()
p.op_type   # "__any__"  (internal sentinel)
```

### `Pattern.const(value)`

Match a **constant tensor / initializer** with the given value. Accepts:
- A scalar (`float`, `int`)
- A `numpy.ndarray`

```python
p = Pattern.const(1.0)          # scalar
p.op_type   # "__const__"
p.value     # 1.0

import numpy as np
B_val = np.array([[1, 2, 3]], dtype=np.float32)
p = Pattern.const(B_val)        # array
```

> Source: `test_pattern.py::test_pattern_dsl`, `test_dsl_deep.py::test_pattern_basics`

---

## Arithmetic Operator Overloading

You can build patterns using Python arithmetic operators — very natural for neural network expressions:

```python
x = Pattern.any()

# Addition
p = x + 1.0             # Pattern.op("Add", x, Pattern.const(1.0))
p = 1.0 + x             # Pattern.op("Add", Pattern.const(1.0), x)

# Multiplication
p = x * 2.0             # Pattern.op("Mul", x, Pattern.const(2.0))

# Division
p = x / 2.0             # Pattern.op("Div", x, Pattern.const(2.0))

# Power
p = x ** 2              # Pattern.op("Pow", x, Pattern.const(2))

# Negation
p = -x                  # Pattern.op("Neg", x)
```

**Composing:**
```python
x = Pattern.any()
p = (x + 1.0) * 2.0     # Mul( Add(any, 1.0), 2.0 )

p.op_type                # "Mul"
p.inputs[0].op_type      # "Add"
p.inputs[1].value        # 2.0
p.inputs[0].inputs[0] is x   # True
p.inputs[0].inputs[1].value  # 1.0
```

**Using a real graph tensor value:**
```python
graph = NeuronGraph("model.onnx")
B_val = graph.tensor_map["B"]   # np.ndarray

x = Pattern.any()
pat = x + B_val          # matches Add(anything, B_initializer)
```

> Source: `test_pattern.py::test_pattern_arithmetic_dsl`, `test_dsl_deep.py::test_pattern_arithmetic`, `comprehensive_test.py::test_pattern_dsl_and_matching`

---

## Sequential Chaining — `.then()`

Chain two patterns in sequence (one feeds into the next):

```python
# Match: Identity → Add  (Identity's output is Add's input)
pat = Pattern.op("Identity").then("Add")

match = graph.match(pat)
match.nodes[0].op_type   # "Identity"
match.nodes[1].op_type   # "Add"
match.start.op_type      # "Add"  (the LAST / root node of the match)
```

> Source: `test_fusion.py::test_sequential_dsl`, `comprehensive_test.py::test_pattern_dsl_and_matching`

---

## Alternative Matching — `Pattern.any_of()`

Match if **any one** of the given alternatives matches:

```python
p = Pattern.any_of(
    Pattern.op("Sigmoid", Pattern.any()),
    Pattern.op("Tanh",    Pattern.any()),
)

# Matches Sigmoid node
res1 = parser.pattern_detect(p, start_node="n4")
res1.start.op_type   # "Sigmoid"

# Also matches Tanh node
res2 = parser.pattern_detect(p, start_node="n5")
res2.start.op_type   # "Tanh"

# Must have at least 2 alternatives:
# Pattern.any_of(p1)   → raises ValueError
```

> Source: `test_pattern.py::test_pattern_any_of`, `test_detector_deep.py::test_detector_any_of`

---

## Constraint Modifiers

These are fluent methods you chain onto any Pattern to add extra constraints:

### `.where(**attrs)` — attribute constraints

```python
# Match Conv with kernel_shape == [3, 3]
p = Pattern.op("Conv").where(kernel_shape=[3, 3])

# Callable constraint (more flexible)
p = Pattern.op("Conv").where(kernel_shape=lambda x: x[0] == 3)
```

### `.with_output_rank(n)` — shape constraint

```python
# Match Mul whose first output has rank 2
p = Pattern.op("Mul").with_output_rank(2)
```

### `.with_dtype(dtype_str)` — data type constraint

```python
# Match Mul producing float32
p = Pattern.op("Mul").with_dtype("float32")
```

### `.capture(name)` — bind the matched node to a name

```python
p = Pattern.op("Add",
    Pattern.any().capture("input_a"),
    Pattern.const(B_val).capture("input_b")
).capture("add_node")

res = parser.pattern_detect(p, start_node="node_add")
res.bindings["add_node"].name    # "node_add"
res.bindings["input_a"].name     # "node_id"
```

> Source: `test_detector_deep.py::test_detector_attribute_constraints`, `test_detector_shape_dtype_constraints`

---

## Built-in Activation Patterns

The Pattern DSL ships with ready-made patterns for common activation functions:

```python
x = Pattern.any()

Pattern.relu(x)        # Pattern.op("Relu", x)
Pattern.sigmoid(x)     # Pattern.op("Sigmoid", x)
Pattern.hardswish(x)   # x * hardsigmoid(x)  → Mul node
Pattern.gelu(x)        # GELU approximation   → Mul node
Pattern.relu6(x)       # Clip(x, min=0, max=6)

p = Pattern.relu6(x)
p.op_type              # "Clip"
p.inputs[1].value      # 0.0
p.inputs[2].value      # 6.0
```

### Mish activation

```python
p = Pattern.mish(x)    # x * tanh(softplus(x))

p.op_type                        # "Mul"
p.inputs[0] is x                 # True
p.inputs[1].op_type              # "Tanh"
p.inputs[1].inputs[0].op_type   # "Softplus"
```

> Source: `test_dsl_deep.py::test_pattern_activations`, `test_pattern_mish`

---

## Calling a Pattern Like a Function

A pattern with no inputs yet can be called with input patterns:

```python
relu = Pattern.op("Relu")
x = Pattern.any()

p = relu(x)            # → Pattern.op("Relu", x)
p.op_type              # "Relu"
p.inputs               # [x]

# If pattern already HAS inputs, calling again raises:
# p(x)   → ValueError: already has inputs
```

> Source: `test_dsl_deep.py::test_pattern_call`

---

## Inspecting a Pattern's Nodes

```python
pat = Pattern.op("Add", Pattern.op("Mul"), Pattern.const(1.0))

nodes = pat.nodes()         # flat list of all Pattern nodes in tree
len(nodes)                  # 3  (Add, Mul, const)

pat.parents[0].op_type      # "Mul"   (first input pattern)
```

> Source: `test_fusion.py::test_pattern_traversal`

---

## What's next?

→ [04 — Pattern Matching](./04_pattern_matching.md) — running patterns against real graphs.
