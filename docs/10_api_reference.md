# 10 — API Reference & Edge Cases

Complete API surface derived from all test cases. Covers every public method, known gotchas, and edge cases.

---

## `NeuronGraph` — Constructor

| Input type | Behaviour |
|---|---|
| `str` ending `.onnx` | ONNX backend |
| `str` ending `.tflite` or `.bin` | TFLite backend |
| `bytes` with `b"TFL3"` at offset 4 | TFLite backend |
| `bytes` without TFL3 magic | Treated as ONNX protobuf |
| `onnx.ModelProto` | ONNX backend |
| `tflite.Model` object | TFLite backend |
| `ONNXParser` | ONNX backend |
| `TFLiteParser` | TFLite backend |

```python
NeuronGraph(path_or_data_or_parser)
NeuronGraph.load(path)   # static — raises ValueError for unknown extension
```

---

## `NeuronGraph` — Properties

| Property | Type | Description |
|---|---|---|
| `graph.nodes` | `list` | All operator nodes |
| `graph.tensor_map` | `LazyTensorMap / dict` | Initializer tensors (lazy on ONNX) |
| `graph.shape_info` | `dict[str, (int\|None, str)]` | `{tensor_name: (rank, dtype)}` |
| `graph._backend` | `ONNXParser / TFLiteParser` | Raw backend parser |
| `graph.passes` | namespace | Built-in optimization passes |

---

## `NeuronGraph` — Methods

| Method | Returns | Description |
|---|---|---|
| `graph.query()` | `NeuronQuery` | Query over all nodes |
| `graph.match(pat)` | `MatchResult\|None` | First pattern match |
| `graph.findall(pat)` | `list[MatchResult]` | All pattern matches |
| `graph.replace(pat, new_op, name=None)` | `NeuronRewriter` | Match-and-replace shortcut |
| `graph.rewriter()` | `NeuronRewriter` | Get blank rewriter |
| `graph.get_weights()` | `dict[str, np.ndarray]` | All initializer tensors as numpy |
| `graph.slice(start, end)` | `NeuronGraph` | Sub-graph between two tensor names/nodes |
| `graph.slice_node(node_or_name)` | `NeuronGraph` | Sub-graph of a single node |
| `graph.save(path)` | `None` | Serialize to file |
| `graph.compare_outputs(...)` | `dict` | Run + compare against PyTorch |
| `graph.verify_against_pytorch(...)` | decorator | Wrap a function with auto-comparison |

---

## `NeuronQuery` — Filter Methods

| Method | Description |
|---|---|
| `.op(op_type)` | Filter by op type (exact) |
| `.name(name, exact=False)` | Filter by node name (substring or exact) |
| `.rank(n)` | Filter by output rank |
| `.dtype(dtype_str)` | Filter by output data type |
| `.attr(key, value=None)` | Filter by attribute (presence or value) |
| `.has_params()` | Only nodes that consume at least one initializer |
| `.filter(predicate)` | Custom callable predicate |
| `.find_by_op_type(op)` | Alias for `.op()` |
| `.find_by_name(name, exact)` | Alias for `.name()` |
| `.find_by_attribute(key)` | Alias for `.attr(key)` |
| `.find_by_tensor(name)` | Nodes producing or consuming a tensor |

---

## `NeuronQuery` — Traversal Methods

| Method | Description |
|---|---|
| `.children()` / `.outputs()` | Immediate downstream nodes |
| `.parents()` / `.inputs()` | Immediate upstream nodes |
| `.descendants()` | All transitive downstream nodes |
| `.ancestors()` | All transitive upstream nodes |
| `.topological_sort()` | Returns sorted query |
| `.is_topologically_sorted()` | `bool` check |

---

## `NeuronQuery` — Set Operations

```python
q1 | q2    # Union
q1 & q2    # Intersection
q1 - q2    # Difference
```

---

## `NeuronQuery` — Accessors

| Method / Property | Description |
|---|---|
| `.count()` / `len(q)` | Number of nodes |
| `.first()` | First node or `None` |
| `.single_node` | Single node (asserts exactly 1) |
| `.tensor()` | `dict[str, np.ndarray]` of parameter tensors |
| `.single_tensor` | Single np.ndarray parameter |
| `q[i]` | `NeuronQuery` containing only the `i`-th node |
| `for n in q:` | Iterate over nodes |
| `.select(pat)` | `list[MatchResult]` from pattern matching |
| `.where(pat)` | `NeuronQuery` filtered by pattern |
| `.follow(pat)` | Follow a structural path pattern from current nodes |
| `.to_pattern()` | Convert query's op type into a `Pattern` |

---

## `Pattern` — Constructors

| Method | Description |
|---|---|
| `Pattern.op(op_type, *inputs)` | Match an operator node |
| `Pattern.any()` | Wildcard — match any node |
| `Pattern.const(value)` | Match a constant/initializer tensor |
| `Pattern.any_of(p1, p2, ...)` | Match any one of ≥2 alternatives |
| `Pattern.relu(x)` | Relu activation pattern |
| `Pattern.sigmoid(x)` | Sigmoid activation pattern |
| `Pattern.hardswish(x)` | HardSwish pattern |
| `Pattern.gelu(x)` | GELU pattern |
| `Pattern.relu6(x)` | ReLU6 (Clip 0–6) |
| `Pattern.mish(x)` | Mish activation |

---

## `Pattern` — Modifier Methods

| Method | Description |
|---|---|
| `.capture(name)` | Name this matched node in `bindings` |
| `.where(**attrs)` | Attribute constraints (value or callable) |
| `.with_output_rank(n)` | Require output rank == n |
| `.with_dtype(dtype_str)` | Require output dtype |
| `.then(op_type)` | Sequential chain to next op |
| `pat(x)` | Call pattern as function to set its inputs |

---

## `Pattern` — Arithmetic Overloading

```python
x = Pattern.any()
x + val     # Add
x * val     # Mul
x / val     # Div
x ** val    # Pow
-x          # Neg
val + x     # Add (reversed)
```

---

## `Pattern` — Matching Methods

| Method | Description |
|---|---|
| `pat.find(graph)` | Returns first `MatchResult` |
| `pat.findall(graph)` | Returns `list[MatchResult]` |
| `pat.filter(graph)` | Returns `NeuronQuery` of matching root nodes |

---

## `MatchResult` — Properties

| Property | Type | Description |
|---|---|---|
| `.start` | node | Root (outermost) matched node |
| `.nodes` | `list` | All matched nodes |
| `.bindings` | `dict[str, node]` | Captured nodes by name |
| `.subgraph` | `NeuronQuery` | Query wrapping matched nodes |
| `.query` | `NeuronQuery` | Query containing only `.start` |

---

## `NeuronRewriter` — Methods

| Method | Description |
|---|---|
| `.replace(nodes, new_op, inputs, outputs, name=None)` | Replace nodes with a new op |
| `.replace_from_result(result, new_op, inputs=None, outputs=None)` | Replace from MatchResult |
| `.delete(nodes)` | Remove nodes |
| `.insert_before(target_node, new_op, inputs, outputs, name=None)` | Insert before node |
| `.register_initializer(name, np_array)` | Add ONNX weight tensor (ONNX only) |
| `.register_tensor(name, shape, dtype, buffer_data=None)` | Add TFLite tensor (TFLite only) |
| `.build(path=None)` | Build new model (returns ModelProto or bytes) |
| `.reset()` | Clear all staged edits |

> **Gotcha:** `.build()` raises `ValueError: No edits staged` if no edits were staged.

---

## `LazyTensorMap` — ONNX Tensor Map

```python
tm = parser.tensor_map     # LazyTensorMap instance

# Dict-like protocol
"B" in tm                  # membership check (no data loaded)
tm["B"]                    # np.ndarray (loads on first access, then caches)
tm.get("B")                # np.ndarray or None
tm.get("x", default)       # default if missing
len(tm)                    # count of initializers
tm.keys()                  # tensor names
tm.values()                # np.ndarrays
tm.items()                 # (name, array) pairs

tm._cache                  # internal cache dict (starts empty)
```

---

## Internal ONNX Utilities

| Function | Description |
|---|---|
| `_attr_value(attr)` | Decode an ONNX AttributeProto to Python value |
| `_node_attrs(node)` | `dict` of all attributes of an ONNX node |
| `_build_shape_info(model)` | `dict[tensor_name, (rank, dtype)]` from a model |
| `_ONNX_DTYPE_TO_NP` | `dict[int, str]` mapping ONNX dtype → numpy dtype string |

---

## Internal TFLite Utilities

```python
from neuron_toolkit.backends.tflite.utils import _get_tflite_attr

attrs = _get_tflite_attr(operator, "CONV_2D")
# Returns dict of parsed attributes, or {} if no options
```

---

## Known Gotchas & Edge Cases

### 1. Non-commutative operators
`Sub`, `Div`, `Pow` etc. — input order **must** match exactly. The detector does NOT try both orderings for these.

### 2. Graph inputs have no producer node
If a pattern specifies `Pattern.op("Relu", Pattern.any())`, the `Pattern.any()` must match a **producer node**. If Relu's input is a raw graph input tensor (e.g. `"A"`), there is no producer node, so the match returns `None`.

### 3. `insert_before` does not rewire
`rewriter.insert_before()` adds a node but does NOT update the target node's inputs. You must additionally call `rewriter.replace()` for the target node with updated input names.

### 4. `rewriter.build()` requires at least one edit
Always stage at least one `replace()`, `delete()`, or `insert_before()` before calling `.build()`.

### 5. Saving a TFLite model loaded from `tflite.Model` object
You cannot call `graph.save()` on a `NeuronGraph` constructed from a raw `tflite.Model` Python object. Load from a file path or bytes instead.

### 6. TFLite bytes must have TFL3 magic
If TFLite bytes do not have the `b"TFL3"` identifier at bytes 4–7, `NeuronGraph` will treat them as ONNX and fail.

### 7. `ONNXParser(infer_shapes=False)`
By default, shape inference is run. You can disable it but `shape_info` may be less complete.

### 8. `PatternDetector` requires `ModelProto`
If using `PatternDetector` directly (not via `NeuronGraph`), pass `parser.model` (the raw `onnx.ModelProto`), not the parser itself.

---

## Public Export Names

```python
import neuron_toolkit

neuron_toolkit.NeuronGraph     # main entry point
neuron_toolkit.NeuronQuery     # query result type
neuron_toolkit.NeuronRewriter  # rewriter type
neuron_toolkit.Pattern         # pattern DSL
```

> Source: `test_multibackend.py::test_neuron_names_export`

---

## Quick Reference Card

```python
from neuron_toolkit import NeuronGraph, Pattern

# Load
g = NeuronGraph("model.onnx")

# Inspect
g.nodes, g.tensor_map, g.shape_info, g.get_weights()

# Query
g.query().op("Conv").has_params().first()
g.query().op("Relu").descendants()
g.query().op("Add") | g.query().op("Mul")

# Pattern
pat = Pattern.op("Conv").then("BatchNormalization")
pat = Pattern.relu(Pattern.any())
pat = (Pattern.any() + 1.0) * 2.0

# Match
g.match(pat)
g.findall(pat)
pat.find(g) ; pat.findall(g) ; pat.filter(g)

# Rewrite
rw = g.replace(pat, "FusedOp")
rw = g.rewriter()
rw.replace([node], "NewOp", inputs=[...], outputs=[...])
rw.delete([node])
rw.insert_before(node, "NewOp", inputs=[...], outputs=[...])
rw.build("output.onnx")

# Built-in passes
g.passes.fuse_conv_bn().build("fused.onnx")

# Slice
g.slice("start_tensor", "end_tensor")
g.slice_node(node)

# Compare
g.compare_outputs(["A"], ["Z"], pytorch_fn, {"A": arr})

@g.verify_against_pytorch(["A"], ["Z"])
def ref(x): return torch.relu(x)
```
