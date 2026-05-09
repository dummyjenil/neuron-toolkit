# ONNX Pattern Detection & Graph Query Engine

A powerful and developer-friendly toolkit for analyzing, searching, traversing, and pattern-matching ONNX computation graphs.

This toolkit provides two core systems:

* **PatternDetector** → Detects mathematical/operator patterns inside ONNX graphs
* **ONNXQuery** → A fluent query engine for exploring ONNX models

Together, they allow you to inspect neural networks like a database and match graph structures like a compiler.

---

# Why This Exists

ONNX models internally are just computation graphs.

A simple operation like:

```text
x * sigmoid(x)
```

actually becomes multiple connected ONNX nodes.

Finding such structures manually is difficult because ONNX graphs can contain:

* thousands of nodes
* nested operator chains
* reused tensors
* optimizer-generated subgraphs
* decomposed activations

This toolkit makes graph inspection simple, readable, and programmable.

---

# What This Toolkit Can Do

With this toolkit you can:

* Find all Conv layers
* Detect GELU / SiLU / Swish activations
* Traverse parent/child nodes
* Search tensors and weights
* Inspect graph topology
* Match complex subgraphs
* Build compiler optimizations
* Create fusion passes
* Analyze quantization-friendly patterns
* Explore neural network architectures

---

# Core Components

| Component         | Purpose                                         |
| ----------------- | ----------------------------------------------- |
| `Pattern`         | Declarative DSL for describing graph structures |
| `PatternDetector` | Recursive graph pattern matching engine         |
| `ONNXQuery`       | Chainable graph query system                    |
| `MatchResult`     | Stores pattern match results                    |
| `_GraphShim`      | Lightweight internal graph abstraction          |

---

# 1. Pattern System

The `Pattern` DSL allows you to describe ONNX subgraphs using mathematical syntax.

Instead of manually checking graph nodes, you write patterns like equations.

---

# Simple Example

```python
x = Pattern.any()

pattern = x * Pattern.sigmoid(x)
```

This describes the famous:

```text
SiLU / Swish activation
```

The detector will automatically search the graph for this structure.

---

# Built-in Activation Patterns

The DSL already includes many common neural network activations.

| Activation   | Helper                 |
| ------------ | ---------------------- |
| ReLU         | `Pattern.relu(x)`      |
| Sigmoid      | `Pattern.sigmoid(x)`   |
| Tanh         | `Pattern.tanh(x)`      |
| SiLU / Swish | `Pattern.silu(x)`      |
| GeLU         | `Pattern.gelu(x)`      |
| Mish         | `Pattern.mish(x)`      |
| HardSwish    | `Pattern.hardswish(x)` |
| Softmax      | `Pattern.softmax(x)`   |
| ReLU6        | `Pattern.relu6(x)`     |

---

# PatternDetector

`PatternDetector` is the engine that actually searches the ONNX graph.

It recursively traverses graph nodes and checks whether a subgraph matches the requested pattern.

---

# How Pattern Matching Works

Suppose your model contains:

```text
Mul
 ├── x
 └── Sigmoid
      └── x
```

The detector understands that this is:

```text
x * sigmoid(x)
```

and matches it as a SiLU activation.

---

# Example Usage

```python
from onnx_toolkit.pattern import Pattern, PatternDetector

x = Pattern.any()

pattern = Pattern.silu(x)

detector = PatternDetector(model)

matches = detector.find_all(pattern)
```

---

# Features of PatternDetector

---

## Recursive Graph Matching

Uses DFS (Depth First Search) to recursively inspect graph structures.

This allows matching:

* shallow patterns
* deep nested subgraphs
* decomposed operators

---

## Wildcard Matching

```python
Pattern.any()
```

matches any node.

Useful for flexible graph rules.

---

## Constant Matching

You can match specific constant values.

```python
Pattern.const(0.5)
```

Supports:

* initializer tensors
* Constant nodes

---

## Alternative Pattern Matching

```python
Pattern.any_of(...)
```

allows multiple valid graph structures.

Example:

```python
Pattern.any_of(
    Pattern.relu(x),
    Pattern.leaky_relu(x)
)
```

---

## Shape & DType Constraints

You can enforce tensor metadata constraints.

```python
Pattern.any()
    .with_output_rank(4)
    .with_dtype("float32")
```

---

## Attribute Constraints

Match ONNX attributes.

```python
Pattern.op("Conv").where(kernel_shape=[3,3])
```

---

## Capture System

Capture matched nodes for later inspection.

```python
x = Pattern.any().capture("input")
```

Later:

```python
result.bindings["input"]
```

---

## Commutative Operator Support

The detector understands that:

```text
a + b
```

is the same as:

```text
b + a
```

for:

* Add
* Mul

---

## Referential Consistency

If the same pattern object appears multiple times:

```python
x = Pattern.any()

pattern = x * x
```

the detector ensures both inputs refer to the exact same node.

---

# MatchResult

Every successful pattern match returns a `MatchResult`.

It contains:

| Property     | Description                  |
| ------------ | ---------------------------- |
| `start`      | Start node of the match      |
| `end`        | End node                     |
| `nodes`      | All matched nodes            |
| `bindings`   | Captured nodes               |
| `as_query()` | Convert match into ONNXQuery |

---

# Example

```python
result.nodes
result.bindings
```

---

# 2. ONNXQuery

`ONNXQuery` is a lazy, chainable graph query engine.

Think of it as:

```text
Pandas + SQL + Graph Traversal
```

for ONNX models.

---

# Why ONNXQuery?

Normally ONNX graphs are difficult to inspect manually.

Without this toolkit you must:

* iterate nodes manually
* resolve tensors yourself
* track parents/children
* manage graph traversal
* handle topology ordering

`ONNXQuery` simplifies all of this.

---

# Basic Query Example

```python
q.op("Conv")
```

Returns all Conv nodes.

---

# Chained Queries

```python
(
    q.op("Conv")
     .children()
     .op("Relu")
)
```

This finds:

```text
Conv → Relu
```

chains.

---

# Features of ONNXQuery

---

## Operator Filtering

```python
q.op("MatMul")
```

---

## Name Search

```python
q.name("attention")
```

---

## Tensor Search

```python
q.find_by_tensor("hidden_states")
```

---

## Attribute Filtering

```python
q.attr("axis", 1)
```

---

## Shape Queries

```python
q.rank(4)
```

---

## DType Queries

```python
q.dtype("float16")
```

---

## Parent Traversal

```python
q.parents()
```

---

## Child Traversal

```python
q.children()
```

---

## Ancestor Traversal

```python
q.ancestors()
```

---

## Descendant Traversal

```python
q.descendants()
```

---

## Pattern Matching Integration

```python
q.matches(pattern)
```

Returns nodes that satisfy a pattern.

---

## Full Match Results

```python
q.match_results(pattern)
```

Returns detailed match objects.

---

## Tensor Access

```python
q.tensor()
```

Access weight tensors directly.

---

## Set Operations

### Union

```python
a | b
```

### Intersection

```python
a & b
```

### Difference

```python
a - b
```

---

## Topological Sorting

```python
q.topological_sort()
```

Returns nodes in execution order.

---

## Functional Pipelines

```python
q.apply(fn)
```

Apply custom logic to matched nodes.

---

# Internal Architecture

The toolkit internally uses:

| Technology | Purpose                    |
| ---------- | -------------------------- |
| `NetworkX` | Graph traversal            |
| `NumPy`    | Tensor comparison          |
| `ONNX`     | Graph parsing              |
| `DFS`      | Recursive pattern matching |
| `BFS`      | Fast graph traversal       |

---

# Performance Features

---

## Cached Graph Structures

Heavy computations are cached using:

```python
@cached_property
```

---

## Optimized Traversal

Uses:

```text
multi-source BFS
```

for fast ancestor/descendant searches.

---

## Safe Backtracking

Pattern matching uses snapshots and rollback logic to safely explore graph branches.

---

## Cycle Handling

Gracefully handles cyclic graphs.

---

# Real-World Use Cases

---

# Compiler Optimizations

Detect:

```text
Conv + BatchNorm + Relu
```

for operator fusion.

---

# Quantization Pipelines

Locate quantization-friendly graph patterns.

---

# Backend Conversion

Prepare models for:

* TensorRT
* TVM
* OpenVINO
* custom runtimes

---

# Neural Network Research

Analyze:

* activation usage
* graph structures
* operator composition

---

# Model Debugging

Trace tensor flow and graph connectivity.

---

# Graph Inspection

Explore unknown ONNX architectures.

---

# Example: Detect All SiLU Activations

```python
from onnx_toolkit.pattern import Pattern
from onnx_toolkit.query import ONNXQuery

x = Pattern.any()

pattern = Pattern.silu(x)

matches = query.match_results(pattern)

for m in matches:
    print(m.nodes)
```

---

# Example: Find All Conv Layers

```python
conv_layers = query.op("Conv")
```

---

# Example: Find Conv → Relu Chains

```python
(
    query
    .op("Conv")
    .children()
    .op("Relu")
)
```

---

# Design Philosophy

This toolkit was designed with these goals:

| Goal                    | Description                                       |
| ----------------------- | ------------------------------------------------- |
| Readability             | Graph queries should look like Python expressions |
| Composability           | Queries should chain naturally                    |
| Declarative APIs        | Describe patterns, not traversal logic            |
| Compiler-style matching | Support advanced graph transformations            |
| Developer productivity  | Reduce ONNX graph boilerplate                     |

---

# Summary

This toolkit transforms ONNX graphs into something that feels easy to search, inspect, and manipulate.

It provides:

* a declarative pattern DSL
* recursive graph pattern matching
* lazy graph querying
* topology traversal
* tensor inspection
* operator filtering
* subgraph analysis
* compiler-grade matching capabilities

In simple words:

```text"
It turns ONNX graphs into a searchable and programmable graph database.
```
