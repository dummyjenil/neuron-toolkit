# 04 — Pattern Matching

Once you have a `Pattern`, you can **detect** it in a graph.  
The result is a `MatchResult` object (or `None` if no match).

---

## Three Ways to Match

| Method | Returns | Description |
|---|---|---|
| `graph.match(pat)` | `MatchResult \| None` | First match anywhere in the graph |
| `graph.findall(pat)` | `list[MatchResult]` | All matches in the graph |
| `parser.pattern_detect(pat, start_node=...)` | `MatchResult \| None` | Match starting from a specific node |

---

## `graph.match()` — First Match

```python
from neuron_toolkit import NeuronGraph, Pattern

graph = NeuronGraph("model.onnx")

# Match any Relu
pat = Pattern.op("Relu")
match = graph.match(pat)

if match:
    print(match.start.op_type)   # "Relu"
```

**Nested patterns** (matches a subgraph rooted at Mul that contains Add as input):

```python
pat = Pattern.op("Mul", Pattern.op("Add"))
match = graph.match(pat)

match.start.op_type      # "Mul"
len(match.nodes)         # 2  (Add, Mul)
match.nodes[0].op_type   # "Add"
match.nodes[1].op_type   # "Mul"
```

> Source: `test_fusion.py::test_onnx_graph_entry_point`, `comprehensive_test.py::test_pattern_dsl_and_matching`

---

## `graph.findall()` — All Matches

```python
# Find all Add nodes
matches = graph.findall(Pattern.op("Add"))
len(matches)                     # 1 (if only 1 Add in graph)

# Each element is a MatchResult
for m in matches:
    print(m.start.op_type, m.start.name)
```

> Source: `test_fusion.py::test_onnx_graph_entry_point`

---

## `parser.pattern_detect()` — Match at a Specific Node

Useful when you already know **which node** is the root of the subgraph you're looking for:

```python
from neuron_toolkit.parser import ONNXParser
from neuron_toolkit.pattern import Pattern

parser = ONNXParser("model.onnx")

pat = Pattern.op("Add", Pattern.any(), Pattern.const(B_val))
res = parser.pattern_detect(pat, start_node="node_add")

if res:
    print(res.start.name)   # "node_add"
```

You can also pass a **node object** instead of a name string:

```python
node = parser.nodes[0]
res = parser.pattern_detect(pat, start_node=node)
```

---

## `MatchResult` — What You Get Back

```python
match = graph.match(Pattern.op("Mul", Pattern.op("Add")))

match.start          # the ROOT node of the match (Mul)
match.nodes          # all nodes in the matched subgraph (Add, Mul)
match.subgraph       # NeuronQuery wrapping those nodes
match.query          # NeuronQuery containing only match.start

# Named bindings (only if you used .capture())
match.bindings       # dict[str, node]
```

### Using `match.query` and `match.subgraph`

```python
match = graph.match(Pattern.op("Add"))
match.query.first().op_type      # "Add"
match.subgraph.count()           # 1
```

> Source: `test_fusion.py::test_query_fluent_fusion`

---

## Named Captures — `match.bindings`

Use `.capture("name")` in the pattern to name parts of the match:

```python
import numpy as np

B_val = np.array([[1, 2, 3]], dtype=np.float32)

pat = Pattern.op(
    "Add",
    Pattern.any().capture("input_a"),
    Pattern.const(B_val).capture("input_b"),
).capture("add_node")

res = parser.pattern_detect(pat, start_node="node_add")

res.bindings["add_node"].name   # "node_add"
res.bindings["input_a"].name    # "node_id"  (the Identity node feeding Add)
```

> Source: `test_pattern.py::test_pattern_capture`

---

## Commutative Operators

`Add` and `Mul` are **commutative** — the detector will try both input orderings automatically.

```python
B_val = np.array([[1, 2, 3]], dtype=np.float32)

# Graph has: Add(Identity_output, B)
# Pattern has inputs in SWAPPED order: Add(const, any)
pat = Pattern.op("Add", Pattern.const(B_val), Pattern.any())

res = parser.pattern_detect(pat, start_node="node_add")
# Still matches! Because Add is commutative.
assert res is not None
```

> Source: `test_pattern.py::test_pattern_commutative`, `test_detector_deep.py::test_detector_commutative`

---

## Non-Commutative Operators — Order Matters!

`Sub`, `Div`, etc. are **NOT** commutative. The input order must match exactly:

```python
# Graph: Sub(Identity(A), B)  where B=1.0
# i.e.  A_id - 1.0

# Correct pattern (1st input = any, 2nd = const 1.0)
pat_correct = Pattern.op("Sub", Pattern.any(), Pattern.const(1.0))
assert parser.pattern_detect(pat_correct, start_node="node_sub") is not None

# WRONG order (1st = const, 2nd = any) — should NOT match
pat_wrong = Pattern.op("Sub", Pattern.const(1.0), Pattern.any())
assert parser.pattern_detect(pat_wrong, start_node="node_sub") is None
```

> Source: `test_bug_non_commutative.py::test_pattern_sub_order`

---

## Attribute Constraints in Matching

```python
# Only match Conv whose kernel_shape == [3, 3]
pat = Pattern.op("Conv").where(kernel_shape=[3, 3])
assert parser.pattern_detect(pat, start_node="conv") is not None

# Wrong attribute value → no match
pat = Pattern.op("Conv").where(kernel_shape=[5, 5])
assert parser.pattern_detect(pat, start_node="conv") is None

# Callable attribute constraint
pat = Pattern.op("Conv").where(kernel_shape=lambda x: x[0] == 3)
assert parser.pattern_detect(pat, start_node="conv") is not None
```

> Source: `test_detector_deep.py::test_detector_attribute_constraints`

---

## Shape & Dtype Constraints in Matching

```python
# Match Mul whose first output has rank=2 and dtype=float32
pat = Pattern.op("Mul").with_output_rank(2).with_dtype("float32")
assert parser.pattern_detect(pat, start_node="n2") is not None

# Rank mismatch → no match
pat = Pattern.op("Mul").with_output_rank(3)
assert parser.pattern_detect(pat, start_node="n2") is None
```

> Source: `test_detector_deep.py::test_detector_shape_dtype_constraints`

---

## Wildcard Matching

`Pattern.any()` matches **any** node:

```python
p = Pattern.any()
# Matches every node
assert parser.pattern_detect(p, start_node="n0") is not None
assert parser.pattern_detect(p, start_node="n1") is not None
```

> Source: `test_detector_deep.py::test_detector_wildcard`

---

## `Pattern.any_of()` in Matching

```python
p = Pattern.any_of(Pattern.op("Add"), Pattern.op("Mul"))

parser.pattern_detect(p, start_node="n1")   # Add → matches
parser.pattern_detect(p, start_node="n2")   # Mul → matches
parser.pattern_detect(p, start_node="n0")   # Relu → None
```

---

## `find_all` via `PatternDetector` Directly

For advanced use, you can use `PatternDetector` to find all matches across the graph without going through `NeuronGraph`:

```python
from neuron_toolkit.pattern import PatternDetector

det = PatternDetector(parser.model)   # pass the ONNX ModelProto

p = Pattern.op("Add")
matches = det.find_all(p)
len(matches)           # 2 (if two Add nodes in graph)
{m.start.name for m in matches}   # {'n1', 'n3'}
```

> Source: `test_detector_deep.py::test_detector_find_all`

---

## Patterns On TFLite Graphs Too

Pattern matching is backend-agnostic. TFLite op names are uppercase (`"ADD"`, `"RELU"`, `"CONV_2D"`, etc.):

```python
from neuron_toolkit.backends.tflite.parser import TFLiteParser

parser = TFLiteParser("model.tflite")

pat = Pattern.op("ADD")
match = parser.pattern_detect(pat, start_node=parser.nodes[0])
match.start.op_type   # "ADD"
```

> Source: `test_tflite_backend.py::test_tflite_pattern_detect`

---

## Pattern.find() / Pattern.findall() / Pattern.filter() — Pattern-native API

You can call matching methods directly on the `Pattern` object:

```python
pat = Pattern.op("Relu")

# First match
match = pat.find(graph)
match.start.op_type   # "Relu"

# All matches
matches = pat.findall(graph)
len(matches)          # 1

# Filter — returns NeuronQuery of matched root nodes
q = pat.filter(graph)
q.count()             # 1
q.first().op_type     # "Relu"
```

> Source: `test_fusion.py::test_pattern_native_fusion`

---

## Common Gotcha — Pattern needs parent nodes

If your pattern says `Pattern.op("Relu", Pattern.any())`, the detector expects Relu to have **at least one input that is another node** (not a graph input). If the input is a raw graph input tensor (no producer node), the match fails:

```python
# n0 is Relu whose input is the graph input "A" (no parent node in graph)
p = Pattern.op("Relu", Pattern.any())
parser.pattern_detect(p, start_node="n0")   # None — no parent node
```

Use `Pattern.op("Relu")` (no input patterns) to match without input constraints.

> Source: `test_detector_deep.py::test_detector_no_match_if_too_few_parents`

---

## What's next?

→ [05 — Graph Rewriting](./05_graph_rewriting.md) — replace, delete, and insert nodes.
