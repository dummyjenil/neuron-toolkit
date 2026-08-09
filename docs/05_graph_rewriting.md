# 05 — Graph Rewriting

The **Rewriter** lets you modify a model's graph: replace nodes, delete nodes, insert new nodes, and add new weight tensors — then materialize everything into a new model.

---

## Getting a Rewriter

```python
from neuron_toolkit import NeuronGraph

graph = NeuronGraph("model.onnx")
rewriter = graph.rewriter()      # NeuronRewriter
```

Or from the lower-level parser:

```python
from neuron_toolkit.parser import ONNXParser

parser = ONNXParser("model.onnx")
rewriter = parser.rewriter()
```

> **Important:** You must stage **at least one edit** before calling `.build()`, otherwise a `ValueError: No edits staged` is raised.

---

## `rewriter.replace()` — Replace Nodes With a New Op

Replace one or more nodes with a single new operator:

```python
# Find the Add node
add_node = graph.query().op("Add").single_node

rewriter.replace(
    [add_node],          # list of nodes to remove
    "Sub",               # new op type
    inputs=list(add_node.input),    # input tensor names for new op
    outputs=list(add_node.output),  # output tensor names for new op
)

new_model = rewriter.build()

op_types = [n.op_type for n in new_model.graph.node]
assert "Sub" in op_types
assert "Add" not in op_types
```

**Fusing multiple nodes into one:**

```python
node_add = parser.find().op("Add").first()
node_mul = parser.find().op("Mul").first()

# Replace both Add+Mul with a single MyFusion op
rewriter.replace(
    nodes=[node_add, node_mul],
    new_op="MyFusion",
    inputs=["A_id", "B", "C"],
    outputs=["output"],
    name="fused_node",          # optional: give the new node a name
)

new_model = rewriter.build()
new_model.graph.node[1].op_type   # "MyFusion"
new_model.graph.node[1].name      # "fused_node"
```

> Source: `test_rewriter.py::test_rewriter_replace`, `test_rewriter_deep.py::test_rewriter_replace`

---

## `rewriter.replace_from_result()` — Replace Using a MatchResult

If you already have a `MatchResult` from pattern matching, use this shortcut:

```python
import numpy as np
from neuron_toolkit.pattern import Pattern

B_val = np.array([[1, 2, 3]], dtype=np.float32)
pat = Pattern.op("Add", Pattern.any(), Pattern.const(B_val))
res = parser.pattern_detect(pat, start_node="node_add")

rewriter = parser.rewriter()
rewriter.replace_from_result(res, "CustomOp")

new_model = rewriter.build()
op_types = [n.op_type for n in new_model.graph.node]
assert "CustomOp" in op_types
assert "Add" not in op_types
```

**With explicit inputs (recommended for multi-node patterns):**

```python
p = Pattern.op("Mul", Pattern.op("Add"))
match = parser.pattern_detect(p, start_node="node_mul")

rewriter.replace_from_result(
    match,
    "FusedAddMul",
    inputs=["A_id", "B", "C"],    # specify inputs explicitly
)
```

> Source: `test_rewriter.py::test_rewriter_replace_from_result`, `test_rewriter_deep.py::test_rewriter_replace_from_result`

---

## `graph.replace()` — High-Level Shortcut

`NeuronGraph.replace()` combines pattern matching + rewriting in one call:

```python
# Replace all Identity nodes with Relu
rewriter = graph.replace(Pattern.op("Identity"), "Relu")
new_model = rewriter.build()

new_graph = NeuronGraph(new_model)
new_graph.query().op("Relu").count()       # 1
new_graph.query().op("Identity").count()   # 0
```

**With a name for the replacement node:**

```python
B_val = graph.tensor_map["B"]
pattern = Pattern.op("Add", Pattern.op("Identity"), Pattern.const(B_val))

rewriter = graph.replace(pattern, "FusedAdd", name="my_fused_add")
rewriter.build("fused_model.onnx")

new_graph = NeuronGraph("fused_model.onnx")
fused = new_graph.query().op("FusedAdd")
fused.count()             # 1
fused.first().name        # "my_fused_add"
```

> Source: `test_unified_graph_dispatch.py::test_graph_replace_shortcut`, `comprehensive_test.py::test_graph_rewriting_and_replacement`

---

## `rewriter.delete()` — Remove Nodes

Remove nodes completely from the graph:

```python
# Delete the Tanh node
tanh_node = graph.query().op("Tanh").single_node

rewriter = graph.rewriter()
rewriter.delete([tanh_node])

new_model = rewriter.build()
assert len(new_model.graph.node) == 5           # was 6
assert "Tanh" not in [n.op_type for n in new_model.graph.node]
```

> Source: `test_rewriter.py::test_rewriter_delete`, `test_rewriter_deep.py::test_rewriter_delete`

---

## `rewriter.insert_before()` — Insert a New Node

Insert a new node **before** an existing target node:

```python
add_node = graph.query().op("Add").single_node

rewriter = graph.rewriter()
rewriter.insert_before(
    add_node,           # insert before this node
    "Identity",         # new op type
    inputs=["A"],       # input tensor names
    outputs=["A_ident"],# output tensor names
    name="new_identity",# optional name
)

new_model = rewriter.build()
assert len(new_model.graph.node) == 4
```

> **Important:** `insert_before` does **not** automatically rewire the target node's inputs. If you want to change what `target_node` consumes, you must also call `rewriter.replace()` for that node with updated inputs.

Example of full rewiring:

```python
rewriter.insert_before(
    node_mul, "Relu",
    inputs=["add_out"],
    outputs=["relu_out"],
    name="new_relu"
)
# Now update node_mul to consume relu_out instead of add_out
rewriter.replace(
    nodes=[node_mul],
    new_op="Mul",
    inputs=["relu_out", "C"],
    outputs=["output"],
    name="node_mul",
)
# Result: Identity → Add → Relu → Mul
```

> Source: `test_rewriter.py::test_rewriter_insert_before`, `test_rewriter_deep.py::test_rewriter_insert_before`

---

## `rewriter.register_initializer()` — Add a New Weight Tensor

Add a new constant weight to the graph (ONNX backend):

```python
import numpy as np

rewriter = graph.rewriter()
rewriter.register_initializer("extra_weight", np.array([3.0], dtype=np.float32))

# Use it as an input to a new node
add_node = graph.nodes[0]
rewriter.replace(
    graph.nodes,
    "Add",
    inputs=["A", "extra_weight"],
    outputs=["Z"],
)

new_model = rewriter.build()
# Verify the initializer was added
names = {i.name for i in new_model.graph.initializer}
assert "extra_weight" in names
assert new_model.graph.node[0].op_type == "Add"
```

> Source: `test_onnx_passes.py::test_register_initializer_adds_constant`

---

## `rewriter.reset()` — Discard All Staged Edits

```python
rewriter.reset()
# All pending edits cleared — starting fresh
# Calling .build() now will raise ValueError: No edits staged
```

> Source: `test_rewriter_deep.py::test_rewriter_reset`

---

## `rewriter.build()` — Materialize the New Model

```python
# Returns a ModelProto (in-memory)
new_model = rewriter.build()

# OR save to disk directly
rewriter.build("output.onnx")   # also returns the ModelProto
```

---

## Manual Rewriting — Full Example

```python
from neuron_toolkit import NeuronGraph

graph = NeuronGraph("simple.onnx")
# Graph structure: Identity(A) → Add(B) → Mul(C) → output
rewriter = graph.rewriter()

add_node = graph.query().op("Add").first()

# Step 1: Insert an Identity before Add
rewriter.insert_before(
    add_node,
    "Identity",
    inputs=["A"],
    outputs=["A_ident_new"],
    name="manual_identity",
)

# Step 2: Build
rewriter.build("manual_rewrite.onnx")

new_graph = NeuronGraph("manual_rewrite.onnx")
assert len(new_graph.nodes) == 4
new_graph.query().name("manual_identity", exact=True).count()  # 1
```

> Source: `comprehensive_test.py::test_manual_rewriter_api`

---

## TFLite Rewriting

The same API works for TFLite, with some extras:

### Replace a TFLite node

```python
from neuron_toolkit.backends.tflite.parser import TFLiteParser

parser = TFLiteParser("model.tflite")
rw = parser.rewriter()
node = parser.nodes[0]

rw.replace([node], "MyFusion", node.input, node.output)
saved_bytes = rw.build("fused.tflite")   # returns bytes AND saves to path

# Verify
new_parser = TFLiteParser("fused.tflite")
new_parser.nodes[0].op_type   # "MyFusion"
```

### `rw.register_tensor()` — register a new TFLite tensor

```python
import numpy as np

data = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
rw.register_tensor(
    "my_const_tensor",    # tensor name
    [3],                  # shape
    "float32",            # dtype string
    buffer_data=data,     # optional: raw bytes for constant tensor
)
```

After building and reloading:

```python
new_parser = TFLiteParser("out.tflite")
assert "my_const_tensor" in new_parser.tensor_map
val = new_parser.tensor_map["my_const_tensor"]
val.shape    # (3,)
val.dtype    # float32
```

### `rw.replace()` with TFLite op options

```python
# Replace with a RESHAPE op and pass shape options
rw.replace([node], "RESHAPE", node.input, node.output, new_shape=[2, 5])

new_parser = TFLiteParser("out.tflite")
new_parser.nodes[0].attrs.get("new_shape")   # [2, 5]
```

> Source: `test_tflite_backend.py::test_tflite_rewriter_functional`, `test_tflite_rewriter_options_serialization`, `test_tflite_rewriter_register_tensor`

---

## What's next?

→ [06 — Built-in Passes](./06_passes.md) — use pre-built graph optimization passes.
