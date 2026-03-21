# GraphRewriter

`GraphRewriter` stages mutations against an `ONNXParser` and applies them atomically when `build()` is called. The original model is never modified.

---

## Obtaining a Rewriter

The recommended way is through `ONNXParser.rewriter()`:

```python
from onnx_toolkit import ONNXParser

parser = ONNXParser("model.onnx")
rw     = parser.rewriter()
```

You can also instantiate it directly:

```python
from onnx_toolkit import GraphRewriter

rw = GraphRewriter(parser)
```

Passing anything other than an `ONNXParser` raises `TypeError`.

---

## Edit Operations

All edit methods return `self` so they can be chained, and all edits are **staged** — nothing is applied to the model until `build()` is called.

### `replace(nodes, new_op, inputs, outputs, name=None, **attrs) → GraphRewriter`

Replace a sequence of nodes with a single new node of type `new_op`.

| Parameter | Type | Description |
|---|---|---|
| `nodes` | `Sequence[NodeProto]` | Nodes to remove (typically `result.nodes` from a `MatchResult`) |
| `new_op` | `str` | ONNX op type of the replacement node |
| `inputs` | `list[str]` | Input tensor names for the new node |
| `outputs` | `list[str]` | Output tensor names for the new node |
| `name` | `str \| None` | Node name (auto-generated as `{new_op}_{n}_rewrite` if omitted) |
| `**attrs` | | ONNX attributes passed to `onnx.helper.make_node` |

```python
x       = Pattern.any().capture("x")
results = parser.find().match_results(Pattern.gelu(x))

for r in results:
    rw.replace(
        nodes   = r.nodes,
        new_op  = "Gelu",
        inputs  = [r.bindings["x"].output[0]],
        outputs = [r.start.output[0]],
    )
```

---

### `replace_from_result(result, new_op, inputs=None, outputs=None, name=None, **attrs) → GraphRewriter`

Convenience wrapper around `replace()` that derives `inputs` and `outputs` directly from the `MatchResult`.

- `inputs` defaults to `[result.end.output[0]]` — the first output of the boundary node (the last node upstream of the matched subgraph).
- `outputs` defaults to `[result.start.output[0]]` — the first output of the match root.

```python
x       = Pattern.any().capture("x")
results = parser.find().match_results(Pattern.gelu(x))

for r in results:
    rw.replace_from_result(r, new_op="Gelu")
```

This is the most concise form for the common fusion pattern.

---

### `delete(nodes: Sequence[NodeProto]) → GraphRewriter`

Remove nodes from the graph without inserting a replacement.

> **Warning:** Deleting nodes without rewiring their consumers will produce a disconnected graph that may fail shape inference or runtime validation. Make sure downstream nodes are also removed or their inputs are rewired.

```python
# Remove all Dropout nodes (safe if downstream nodes accept the pass-through tensor)
dropout_nodes = parser.find().find_by_op_type("Dropout").nodes
rw.delete(dropout_nodes)
```

---

### `insert_before(target_node, new_op, inputs, outputs, name=None, **attrs) → GraphRewriter`

Insert a new node into the graph. The new node's `outputs` are expected to be connected to `target_node`'s inputs by the caller.

`insert_before` does not automatically rewire any edges — it simply schedules the new node for insertion. You must ensure that `outputs` matches whatever `target_node` now consumes.

```python
target = parser.find().find_by_op_type("Sigmoid").single_node

rw.insert_before(
    target_node = target,
    new_op      = "Relu",
    inputs      = ["add_out"],        # existing tensor
    outputs     = ["relu_pre_sig"],   # new tensor name
)
# You are also responsible for ensuring that target_node
# now reads "relu_pre_sig" instead of "add_out".
```

---

### `reset() → GraphRewriter`

Discard all pending edits. Returns `self`.

```python
rw.replace(nodes, "Gelu", inputs, outputs)
# ... decide not to proceed
rw.reset()
# rw is now clean; build() would raise ValueError
```

---

## Building the Rewritten Model

### `build(output_path: str | None = None) → onnx.ModelProto`

Apply all staged edits and return the rewritten model as a `ModelProto`.

**Build order:**

1. All nodes scheduled for removal are dropped from the graph.
2. All scheduled insertions are appended at the end of the node list.
3. Shape inference is re-run on the result. If it fails, a warning is logged but the model is still returned.
4. IR version and opset imports are copied from the original model.

If `output_path` is given, the model is also saved to that path.

Raises `ValueError` if no edits have been staged.

```python
new_model = rw.build("model_optimised.onnx")
print(f"Rewritten model has {len(new_model.graph.node)} nodes")
```

---

## Chaining Multiple Edit Types

All edit methods return `self`, so multiple operations can be chained in a single expression:

```python
rw = (
    parser.rewriter()
    .delete(parser.find().find_by_op_type("Dropout").nodes)
    .delete(parser.find().find_by_op_type("Identity").nodes)
)
rw.build("cleaned.onnx")
```

You can also apply multiple `replace` calls — one per pattern match — before calling `build()` once:

```python
rw = parser.rewriter()

x = Pattern.any().capture("x")

for r in parser.find().match_results(Pattern.gelu(x)):
    rw.replace_from_result(r, new_op="Gelu")

for r in parser.find().match_results(Pattern.silu(x)):
    rw.replace_from_result(r, new_op="HardSwish")  # example substitution

rw.build("multi_fused.onnx")
```

---

## Passing Attributes to New Nodes

Any keyword argument passed to `replace()`, `replace_from_result()`, or `insert_before()` (beyond the named parameters) is forwarded as an ONNX node attribute via `onnx.helper.make_node`:

```python
# Insert a Clip node with min/max attributes
rw.insert_before(
    target_node = some_node,
    new_op      = "Clip",
    inputs      = ["feature"],
    outputs     = ["clipped"],
    min         = 0.0,
    max         = 6.0,
)
```

Attribute values must be types accepted by `onnx.helper.make_node` (floats, ints, strings, lists of those, or numpy arrays).

---

## Important Limitations

**Node ordering.** Inserted nodes are appended at the end of the node list before `build()` runs shape inference. ONNX does not require topological ordering in the proto, and most runtimes handle arbitrary ordering, but some strict validators may complain. If needed, load the built model with `ONNXParser` and call `parser.find().topological_sort().to_onnx(path)` to re-emit it in topological order.

**Single-node replacements.** Each `replace()` call produces exactly one new node. If you need to replace a subgraph with multiple new nodes, call `replace()` multiple times with carefully chosen input/output tensor names.

**Tensor dtype annotations.** The rewriter does not update `value_info` for new tensors introduced by inserted or replacement nodes. Shape inference (run automatically in `build()`) will populate these where possible.

**No edge rewiring.** The rewriter does not automatically update references to tensors produced by deleted nodes. If you delete a node whose output `"foo"` is consumed by a node you are keeping, you must ensure the kept node's input is updated. This is best achieved by using `replace()` — which schedules an exact replacement with compatible outputs — rather than bare `delete()`.
