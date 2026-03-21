# onnx_toolkit

A Python library for loading, querying, pattern-matching, and rewriting ONNX computation graphs. It provides a fluent query API, a structural pattern DSL, and a staged graph rewriter that compose into concise analysis and transformation pipelines.

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Components](#components)
  - [ONNXParser](#onnxparser)
  - [ONNXQuery](#onnxquery)
  - [Pattern DSL](#pattern-dsl)
  - [PatternDetector & MatchResult](#patterndetector--matchresult)
  - [GraphRewriter](#graphrewriter)
- [Examples](#examples)
  - [Inspect a model](#inspect-a-model)
  - [Query and traverse the graph](#query-and-traverse-the-graph)
  - [Match structural patterns](#match-structural-patterns)
  - [Rewrite the graph](#rewrite-the-graph)
  - [Export subgraphs](#export-subgraphs)
- [Architecture](#architecture)
- [Documentation](#documentation)

---

## Features

- **Fluent query API** — chain filters, traversals, and set operations on graph nodes without mutating state.
- **Pattern DSL** — describe subgraph structures with Python arithmetic operators, named captures, attribute constraints, shape and dtype constraints, and union patterns.
- **Full match results** — every match returns the visited subgraph, named bindings, start node, and boundary node, not just a boolean.
- **Staged rewriter** — accumulate replacements, deletions, and insertions and apply them atomically in a single `build()` call. The original model is never modified.
- **Export** — save any selected subgraph to `.onnx`, Graphviz DOT, or a pandas DataFrame.
- **Shape inference** — runs automatically on load; rank and dtype constraints in patterns work out of the box.
- **18 activation helpers** — built-in patterns for GELU, SiLU, Mish, GELU-tanh, ReLU6, and more, each expanding to the full multi-node subgraph they are typically lowered to.

---

## Installation

Requires Python 3.8+.

```bash
pip install onnx numpy
pip install pandas          # optional — needed for .to_dataframe()
```

Then place the `onnx_toolkit` package directory on your Python path, or install as a local package:

```bash
pip install onnx_toolkit
```

---

## Quick Start

```python
from onnx_toolkit import ONNXParser, Pattern

parser = ONNXParser("model.onnx")
print(parser.summary())

# --- Query ---
convs = parser.find().find_by_op_type("Conv").has_params()
print(convs.count(), "Conv nodes with weights")

# --- Pattern match ---
x       = Pattern.any().capture("x")
results = parser.find().match_results(Pattern.gelu(x))

for r in results:
    print(f"GELU at {r.start.name}, input: {r.bindings['x'].name}")
    print(f"  subgraph: {[n.name for n in r.nodes]}")

# --- Rewrite ---
rw = parser.rewriter()
for r in results:
    rw.replace_from_result(r, new_op="Gelu")
rw.build("model_fused.onnx")
```

---

## Components

### ONNXParser

Loads a `.onnx` file and exposes the graph for querying and rewriting. Shape inference runs automatically on load.

```python
from onnx_toolkit import ONNXParser

parser = ONNXParser("model.onnx")               # shape inference on by default
parser = ONNXParser("model.onnx", infer_shapes=False)

print(parser.summary())

# Direct access to graph data
print(len(parser.nodes))                        # total node count
print(parser.tensor_map["conv1.weight"].shape)  # weight tensor by name
print(parser.graph_inputs)                      # set of input tensor names
print(parser.shape_info.get("relu_0"))          # (rank, dtype) → (4, "float32")
```

---

### ONNXQuery

A lazy, chainable view over a subset of graph nodes. Every method returns a new `ONNXQuery` without mutating the receiver.

```python
q = parser.find()   # start with all nodes

# Filters
q.find_by_op_type("Conv")
q.find_by_name("layer1")                              # substring
q.find_by_name("/layer1/conv1", exact=True)
q.find_by_attribute("group", lambda g: g > 1)         # callable predicate
q.find_by_param_name("bias")
q.has_params()

# Traversal
q.children()
q.parents()
q.ancestors(max_depth=5)
q.descendants(max_depth=3)
q.entry_nodes        # nodes that read graph-level inputs
q.output_nodes       # nodes that write graph-level outputs

# Ordering
q.topological_sort()
q.is_topologically_sorted()

# Weight tensors
q.tensor("conv1.weight")          # single array by name
q.find_by_op_type("Conv")[0].tensor()  # {name: array} for one node
q.find_by_op_type("Conv").tensor()     # {node_name: {name: array}} for many

# Functional
q.apply(lambda node, params: print(node.name, params))

# Set operations
a.union(b)
a.intersection(b)
a.difference(b)

# Export
q.to_dot(show_params=True)       # Graphviz DOT string
q.to_dataframe()                  # pandas DataFrame
q.to_onnx("subgraph.onnx")       # standalone ONNX file

# Iteration
for node in q: ...
q[0]          # first node as ONNXQuery
q[0:3]        # slice
len(q)        # node count
bool(q)       # True if non-empty
```

---

### Pattern DSL

Patterns describe the shape of a subgraph using Python operators and method chaining. They are independent of any model and can be reused across many matches.

**Primitives**

```python
Pattern.any()            # wildcard — matches any node
Pattern.const(0.5)       # constant tensor ≈ 0.5 (tolerance 1e-3)
Pattern.op("MatMul")     # node with this op type
Pattern.any_of(p1, p2)   # union — first alternative that matches wins
```

**Arithmetic operators** map directly to ONNX ops:

```python
x = Pattern.any()

x + Pattern.const(1.0)   # Add
x * 0.5                  # Mul  (numbers auto-coerced to const)
x / Pattern.const(1.414) # Div
x ** 3.0                 # Pow
x - Pattern.const(1.0)   # Sub
-x                       # Neg
Pattern.op("Erf")(x)     # unary application
```

`Add` and `Mul` are matched **commutatively** — input order in the graph does not matter.

**Modifiers**

```python
Pattern.any().capture("x")                          # bind to MatchResult.bindings["x"]
Pattern.op("Conv").where(group=lambda g: g > 1)     # attribute predicate
Pattern.op("Conv").where(kernel_shape=[3, 3])       # attribute exact match
Pattern.op("MatMul").with_output_rank(2)            # shape constraint
Pattern.op("Conv").with_dtype("float16")            # dtype constraint
```

All modifiers return a new `Pattern` and can be chained in any order.

**Built-in activation helpers**

Each helper expands to the full multi-node subgraph that activation is typically lowered to in ONNX:

| Helper | ONNX subgraph |
|---|---|
| `Pattern.relu(x)` | `Relu(x)` |
| `Pattern.sigmoid(x)` | `Sigmoid(x)` |
| `Pattern.tanh(x)` | `Tanh(x)` |
| `Pattern.leaky_relu(x)` | `LeakyRelu(x)` |
| `Pattern.elu(x)` | `Elu(x)` |
| `Pattern.selu(x)` | `Selu(x)` |
| `Pattern.hardsigmoid(x)` | `HardSigmoid(x)` |
| `Pattern.hardswish(x)` | `x * HardSigmoid(x)` |
| `Pattern.silu(x)` / `Pattern.swish(x)` | `x * Sigmoid(x)` |
| `Pattern.gelu(x)` | `x * (Erf(x / √2) + 1) * 0.5` |
| `Pattern.gelu_tanh(x)` | tanh approximation |
| `Pattern.mish(x)` | `x * Tanh(Softplus(x))` |
| `Pattern.relu6(x)` | `Clip(x, 0, 6)` |
| `Pattern.softmax(x)` | `Softmax(x)` |
| `Pattern.log_softmax(x)` | `LogSoftmax(x)` |
| `Pattern.prelu(x, slope)` | `PRelu(x, slope)` |

---

### PatternDetector & MatchResult

`PatternDetector` runs a DFS upstream from a start node and returns a `MatchResult` on success.

```python
from onnx_toolkit import PatternDetector, Pattern

x   = Pattern.any().capture("x")
det = PatternDetector(parser.model, start_node="Mul_42")
r   = det.match(Pattern.silu(x))

if r:
    print(r.start.name)          # "Mul_42"
    print(r.end.name)            # boundary node (or start if no end_node given)
    print(r.nodes)               # all visited NodeProtos, deduplicated
    print(r.bindings["x"].name)  # the node that matched the "x" capture
    sub = r.as_query()           # convert to ONNXQuery for further analysis
```

**`end_node`** is an exclusive upstream boundary — the DFS stops when it would enter that node, treating the branch as satisfied. The matched subgraph is everything between `start_node` (inclusive) and `end_node` (exclusive):

```python
# Match the SiLU at Mul_7 but don't cross Add_3
det = PatternDetector(parser.model, start_node="Mul_7", end_node="Add_3")
r   = det.match(Pattern.silu(Pattern.any()))
# r.end.name == "Add_3"
```

For scanning all nodes, prefer `ONNXQuery.match_results()` which shares the output→node map across all candidates:

```python
results = parser.find().match_results(Pattern.gelu(x))  # efficient bulk scan
```

---

### GraphRewriter

Stages graph mutations and applies them atomically. The original model is never modified until `build()` is called.

```python
rw = parser.rewriter()

# Replace a matched subgraph with a single new op
for r in parser.find().match_results(Pattern.gelu(x)):
    rw.replace_from_result(r, new_op="Gelu")

# Or specify inputs/outputs explicitly
rw.replace(
    nodes   = r.nodes,
    new_op  = "Gelu",
    inputs  = [r.bindings["x"].output[0]],
    outputs = [r.start.output[0]],
)

# Delete nodes entirely
rw.delete(parser.find().find_by_op_type("Dropout").nodes)

# Insert a new node before an existing one
rw.insert_before(target_node, "Relu", inputs=["feat"], outputs=["feat_relu"])

# Discard all pending edits
rw.reset()

# Apply everything and save
new_model = rw.build("model_optimised.onnx")
```

All edit methods return `self` and can be chained. Shape inference is re-run automatically after `build()`.

---

## Examples

### Inspect a model

```python
parser = ONNXParser("model.onnx")
print(parser.summary())

# Op type frequency
from collections import Counter
counts = Counter(n.op_type for n in parser.nodes)
for op, n in counts.most_common(5):
    print(f"  {op:<25} {n}")

# Total parameter count
total = sum(v.size for v in parser.tensor_map.values())
print(f"Parameters: {total:,}")
```

---

### Query and traverse the graph

```python
# Depthwise convolutions without bias
dw    = parser.find().find_by_attribute("group", lambda g: g > 1)
nobias = dw.difference(dw.find_by_param_name("bias"))
print(f"{nobias.count()} depthwise Conv nodes without bias")

# What feeds into every BatchNorm?
bn_parents = parser.find().find_by_op_type("BatchNormalization").parents()
print(bn_parents.op_types())

# Export the first five layers as a standalone model
early = parser.find().entry_nodes.descendants(max_depth=4)
early.union(parser.find().entry_nodes).to_onnx("early.onnx")

# Visualise
with open("graph.dot", "w") as f:
    f.write(parser.find().to_dot(show_params=True))
# dot -Tsvg graph.dot -o graph.svg
```

---

### Match structural patterns

```python
x = Pattern.any().capture("x")

# Any activation function
any_act = Pattern.any_of(
    Pattern.relu(x), Pattern.gelu(x), Pattern.silu(x),
    Pattern.tanh(x), Pattern.sigmoid(x),
)
parser.find().matches(any_act).apply(lambda n, _: print(n.op_type, n.name))

# Depthwise conv → BatchNorm pairs with captures
dw = Pattern.op("Conv").where(group=lambda g: g > 1).capture("conv")
for r in parser.find().find_by_op_type("BatchNormalization").match_results(
    Pattern.op("BatchNormalization", dw)
):
    print("conv:", r.bindings["conv"].name)
    print("bn:  ", r.start.name)

# fp16 MatMul only
fp16_mm = Pattern.op("MatMul").with_dtype("float16").capture("mm")
for r in parser.find().match_results(fp16_mm):
    print("fp16 MatMul:", r.bindings["mm"].name)
```

---

### Rewrite the graph

```python
x  = Pattern.any().capture("x")
rw = parser.rewriter()

# Fuse all activation subgraphs in one pass
fusions = {
    Pattern.gelu(x):      "Gelu",
    Pattern.gelu_tanh(x): "Gelu",
    Pattern.silu(x):      "Silu",
    Pattern.mish(x):      "Mish",
}

total = 0
for pat, new_op in fusions.items():
    for r in parser.find().match_results(pat):
        rw.replace_from_result(r, new_op=new_op)
        total += 1

rw.delete(parser.find().find_by_op_type("Dropout").nodes)

new_model = rw.build("model_optimised.onnx")
print(f"Fused {total} activations, removed Dropout")
print(f"Nodes: {len(parser.nodes)} → {len(new_model.graph.node)}")
```

---

### Export subgraphs

```python
# Save matched subgraphs to individual files
for i, r in enumerate(parser.find().match_results(Pattern.gelu(x))):
    r.as_query().to_onnx(f"gelu_{i}.onnx")
    r.as_query().to_dot()

# Inspect as a DataFrame
df = parser.find().find_by_op_type("Conv").to_dataframe()
print(df[["name", "num_params", "param_shapes"]])
```

---

## Architecture

```
ONNXParser
│
├── .find()  ─────────────────────────────► ONNXQuery
│                                           │
│                                           ├── .matches(pattern)       ─► ONNXQuery
│                                           ├── .match_results(pattern) ─► List[MatchResult]
│                                           ├── .topological_sort()     ─► ONNXQuery
│                                           ├── .apply(fn)              ─► ONNXQuery
│                                           ├── .to_dot()               ─► str
│                                           ├── .to_dataframe()         ─► DataFrame
│                                           └── .to_onnx(path)
│
├── .pattern_detect(pattern, start, end) ─► MatchResult | None
│
└── .rewriter() ──────────────────────────► GraphRewriter
                                            │
                                            ├── .replace(nodes, new_op, …)
                                            ├── .replace_from_result(result, new_op)
                                            ├── .delete(nodes)
                                            ├── .insert_before(target, new_op, …)
                                            ├── .reset()
                                            └── .build(path?) ──────────► ModelProto
```

**Package layout**

```
onnx_toolkit/
├── __init__.py    Public API: ONNXParser, ONNXQuery, Pattern,
│                  PatternDetector, MatchResult, GraphRewriter
├── parser.py      ONNXParser
├── query.py       ONNXQuery
├── pattern.py     Pattern, PatternDetector, MatchResult
├── rewriter.py    GraphRewriter
├── _utils.py      Internal helpers (attr extraction, shape info, graph shim)
└── _types.py      TensorMap type alias
```

---

## Documentation

Full reference documentation is in the `docs/` directory:

| File | Contents |
|---|---|
| [`docs/getting_started.md`](docs/getting_started.md) | Installation, first steps, logging |
| [`docs/parser.md`](docs/parser.md) | `ONNXParser` — all attributes and methods |
| [`docs/query.md`](docs/query.md) | `ONNXQuery` — all filters, traversal, export |
| [`docs/pattern.md`](docs/pattern.md) | `Pattern` DSL — all constructors, modifiers, operators |
| [`docs/detector.md`](docs/detector.md) | `PatternDetector` and `MatchResult` — matching semantics |
| [`docs/rewriter.md`](docs/rewriter.md) | `GraphRewriter` — all edit operations and limitations |
| [`docs/cookbook.md`](docs/cookbook.md) | 60+ worked examples across 11 categories |

---

## Logging

`onnx_toolkit` uses Python's standard `logging` module. To enable debug output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Individual sub-loggers for targeted debugging:

| Logger | Covers |
|---|---|
| `onnx_toolkit` | Parser load and summary |
| `onnx_toolkit.query` | Filter and traversal steps |
| `onnx_toolkit.pattern` | Pattern construction |
| `onnx_toolkit.detect` | DFS walk, decisions, and backtracking |
| `onnx_toolkit.rewriter` | Staged edits and build steps |