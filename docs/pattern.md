# Pattern DSL

`Pattern` is a DSL for describing the *shape* of an ONNX subgraph. Patterns are independent of any specific model — the same pattern object can be matched against many different models or many different nodes within the same model.

---

## Core Concepts

A `Pattern` describes a node and, optionally, its upstream parents recursively. The DFS match walks *upstream* (towards inputs), so a pattern rooted at node `X` describes `X` and all the nodes that feed into it.

Three primitive patterns:

| Pattern | Meaning |
|---|---|
| `Pattern.any()` | Match any single node regardless of op type |
| `Pattern.const(v)` | Match a constant tensor with value ≈ `v` (tolerance 1e-3) |
| `Pattern.op("Add")` | Match a node whose `op_type` is `"Add"` |

---

## Named Constructors

### `Pattern.any() → Pattern`

Wildcard. Matches any node unconditionally. Often used as a placeholder for the "input" to a pattern.

```python
x = Pattern.any()
p = Pattern.relu(x)   # Relu whose input can be anything
```

---

### `Pattern.const(value) → Pattern`

Match a constant tensor. Checks initializer inputs of a node and `Constant` op nodes. Comparison uses `np.allclose(actual, value, atol=1e-3)` so small floating-point differences are tolerated.

```python
half    = Pattern.const(0.5)
sqrt2   = Pattern.const(1.41421356237)
zero    = Pattern.const(0.0)
```

---

### `Pattern.op(op_type: str, *input_patterns: Pattern) → Pattern`

Match a node with the given `op_type`. Optionally constrain the node's parent inputs.

```python
# Match any MatMul node
Pattern.op("MatMul")

# Match a MatMul whose first input is anything and second input is a constant
Pattern.op("MatMul", Pattern.any(), Pattern.const(1.0))
```

---

### `Pattern.any_of(*alternatives: Pattern) → Pattern`

Union pattern. Succeeds if **any** of the alternatives matches. The first successful alternative wins and its bindings are kept. Requires at least 2 alternatives.

```python
x   = Pattern.any()
act = Pattern.any_of(
    Pattern.relu(x),
    Pattern.gelu(x),
    Pattern.silu(x),
)
```

---

## Modifiers

Modifiers return a **new** `Pattern` and never mutate the original. They can be chained in any order.

### `.capture(name: str) → Pattern`

Bind the matched node to `name` in `MatchResult.bindings`. Use this to extract specific nodes from a match result.

```python
x    = Pattern.any().capture("input")
gate = Pattern.op("Sigmoid").capture("gate")

p = x * gate
# After matching, result.bindings["input"] and result.bindings["gate"]
# will be the specific NodeProtos that matched.
```

Captures work on any pattern type including wildcards, op patterns, and union patterns.

---

### `.where(**constraints) → Pattern`

Add attribute constraints. Each keyword is an ONNX attribute name. Values may be:

- A **literal** — checked with `==`.
- A **callable** `fn(attr_value) -> bool` — node passes if the callable returns `True`.

A node fails the constraint if the attribute is **absent**, regardless of the expected value.

```python
# Exact value
Pattern.op("Gather").where(axis=0)

# Callable predicate
Pattern.op("Conv").where(group=lambda g: g > 1)

# Multiple constraints (all must pass)
Pattern.op("Conv").where(
    group=lambda g: g > 1,
    dilations=lambda d: all(v == 1 for v in d),
)
```

---

### `.with_output_rank(rank: int) → Pattern`

Require the node's **first output** tensor to have the given rank. Requires shape inference to have run (`ONNXParser(infer_shapes=True)`, which is the default).

```python
# Match only 4-D (NCHW) Conv nodes
Pattern.op("Conv").with_output_rank(4)

# Match 2-D MatMul outputs
Pattern.op("MatMul").with_output_rank(2)
```

---

### `.with_dtype(dtype: str) → Pattern`

Require the node's **first output** tensor to have the given numpy dtype string. Requires shape inference.

Supported strings: `"float32"`, `"float16"`, `"float64"`, `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`, `"uint16"`, `"uint32"`, `"uint64"`, `"bool"`, `"object"`.

```python
# Only match fp16 operations
Pattern.op("MatMul").with_dtype("float16")

# Match int8 quantised convolutions
Pattern.op("Conv").with_dtype("int8")
```

---

## Arithmetic DSL

Arithmetic operators build compound patterns using standard Python syntax:

| Operator | ONNX op | Example |
|---|---|---|
| `a + b` | `Add` | `x + Pattern.const(1.0)` |
| `a - b` | `Sub` | `x - Pattern.const(0.0)` |
| `a * b` | `Mul` | `x * Pattern.sigmoid(x)` |
| `a / b` | `Div` | `x / Pattern.const(1.414)` |
| `a ** b` | `Pow` | `x ** Pattern.const(3.0)` |
| `-a` | `Neg` | `-x` |

`Add` and `Mul` patterns are matched **commutatively** — both `x + 1` and `1 + x` will match an `Add` node whose inputs are `x` and a constant `1`, regardless of which comes first in the ONNX graph.

Calling a pattern as a function applies it as a **unary wrapper**:

```python
erf = Pattern.op("Erf")
p   = erf(x)   # Erf node whose input matches x
```

Raw Python numbers are automatically coerced to `Pattern.const(...)`:

```python
x * 0.5      # equivalent to x * Pattern.const(0.5)
x + 1.0      # equivalent to x + Pattern.const(1.0)
x ** 3.0     # equivalent to x ** Pattern.const(3.0)
```

---

## Activation Function Helpers

All helpers accept a `Pattern` argument representing the activation's input. They expand into the full multi-node subgraph the activation is typically implemented as in ONNX.

| Helper | Expansion |
|---|---|
| `Pattern.relu(x)` | `Relu(x)` |
| `Pattern.sigmoid(x)` | `Sigmoid(x)` |
| `Pattern.tanh(x)` | `Tanh(x)` |
| `Pattern.leaky_relu(x)` | `LeakyRelu(x)` |
| `Pattern.elu(x)` | `Elu(x)` |
| `Pattern.selu(x)` | `Selu(x)` |
| `Pattern.softplus(x)` | `Softplus(x)` |
| `Pattern.softsign(x)` | `Softsign(x)` |
| `Pattern.hardsigmoid(x)` | `HardSigmoid(x)` |
| `Pattern.hardswish(x)` | `x * HardSigmoid(x)` |
| `Pattern.silu(x)` | `x * Sigmoid(x)` |
| `Pattern.swish(x)` | `x * Sigmoid(x)` (alias for silu) |
| `Pattern.gelu(x)` | `x * (Erf(x / √2) + 1) * 0.5` |
| `Pattern.gelu_tanh(x)` | `0.5 * x * (1 + Tanh(0.7978… * (x + 0.044715 * x³)))` |
| `Pattern.mish(x)` | `x * Tanh(Softplus(x))` |
| `Pattern.relu6(x)` | `Clip(x, 0.0, 6.0)` |
| `Pattern.softmax(x)` | `Softmax(x)` |
| `Pattern.log_softmax(x)` | `LogSoftmax(x)` |
| `Pattern.prelu(x, slope)` | `PRelu(x, slope)` |
| `Pattern.thresholded_relu(x)` | `ThresholdedRelu(x)` |

---

## Combining Modifiers

All modifiers compose freely. The order does not matter semantically:

```python
x = Pattern.any().capture("x")

# A 4-D fp32 depthwise conv whose input is captured
dw = (
    Pattern.op("Conv")
    .where(group=lambda g: g > 1)
    .with_output_rank(4)
    .with_dtype("float32")
    .capture("dw_conv")
)

# Match either depthwise conv or standard conv with captured input
p = Pattern.any_of(dw, Pattern.op("Conv").capture("std_conv"))
```

---

## Patterns Are Immutable

Every modifier returns a **new** `Pattern`. The original is never modified. This means you can safely reuse base patterns:

```python
x   = Pattern.any().capture("x")
dw  = Pattern.op("Conv").where(group=lambda g: g > 1)

# Both of these are independent patterns that share x
p1 = Pattern.relu(dw(x))          # this doesn't work — dw needs to be called like op
p1 = Pattern.relu(x)
p2 = Pattern.gelu(x)
any_act = Pattern.any_of(p1, p2)  # x is safely shared
```

---

## Pattern Matching Semantics

### Constant inputs

Constants in patterns are matched directly against a node's **initializer edges** (weight tensors), not against parent nodes. A pattern like `x * Pattern.const(0.5)` succeeds if the `Mul` node has an initializer input whose value is ≈ 0.5, without requiring a separate `Constant` node in the graph.

### Commutativity

`Add` and `Mul` nodes are matched commutatively. All permutations of parent nodes are tried. For all other ops, parent nodes are matched in the order they appear in the node's input list.

### Constant op nodes

`Pattern.const(v)` also matches a standalone `Constant` op node whose `value` attribute is a tensor ≈ `v`. This handles exporters that emit constants as ops rather than initializers.

### Tolerance

All constant comparisons use `np.allclose(actual, expected, atol=1e-3)`. This accommodates minor floating-point differences between exporters.
