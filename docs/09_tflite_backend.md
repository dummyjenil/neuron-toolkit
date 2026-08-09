# 09 — TFLite Backend

Neuron-toolkit supports TFLite models as a first-class backend. The API is nearly identical to ONNX — the same `NeuronGraph`, `NeuronQuery`, `Pattern`, and rewriter objects work on both.

---

## Loading TFLite Models

```python
from neuron_toolkit import NeuronGraph
from neuron_toolkit.backends.tflite.parser import TFLiteParser

# From a .tflite file path
graph = NeuronGraph("model.tflite")

# From raw bytes (must have b"TFL3" file identifier at bytes 4–7)
tflite_bytes = open("model.tflite", "rb").read()
graph = NeuronGraph(tflite_bytes)

# From a tflite.Model Python object
import tflite
model_obj = tflite.Model.GetRootAsModel(tflite_bytes, 0)
graph = NeuronGraph(model_obj)

# From a TFLiteParser directly
graph = NeuronGraph(TFLiteParser("model.tflite"))

# Using the low-level parser alone
parser = TFLiteParser("model.tflite")
parser.nodes[0].op_type   # "ADD", "ABS", "CONV_2D", etc. (uppercase)
```

> Source: `test_unified_graph_dispatch.py`, `test_tflite_backend.py::test_tflite_parser_bytes_and_object`

---

## TFLite Op Type Names

TFLite uses **uppercase** op names:

| Operation | Op type string |
|---|---|
| Addition | `"ADD"` |
| Absolute value | `"ABS"` |
| Subtraction | `"SUB"` |
| 2D Convolution | `"CONV_2D"` |
| Softmax | `"SOFTMAX"` |
| Reshape | `"RESHAPE"` |
| ReLU | `"RELU"` |

---

## Shape Info from TFLite

```python
graph = NeuronGraph(tflite_abs_bytes)

# shape_info[tensor_name] → (rank, dtype_str)
rank, dtype = graph.shape_info["input"]
rank    # 1  (number of dimensions)
dtype   # "float32"
```

> Source: `test_unified_graph_dispatch.py::test_graph_from_tflite_bytes`

---

## Querying TFLite Nodes

All query operations work identically:

```python
parser = TFLiteParser("model.tflite")
q = parser.find()

q.op("ADD").count()    # 1

# Attribute filtering
q.attr("stride", 1).count()    # nodes with stride attribute == 1
q.attr("stride", 2).count()    # 0

# Fluent chaining
q.op("ADD").has_params().count()  # nodes of type ADD that have constant inputs
```

> Source: `test_multibackend.py::test_tflite_backend_mock`

---

## Attribute Parsing

TFLite operators can have rich built-in options. Neuron-toolkit parses them automatically into the node's `.attrs` dict.

### Conv2D Options

```python
# Attributes parsed for CONV_2D:
node.attrs["stride_w"]                   # int
node.attrs["stride_h"]                   # int
node.attrs["padding"]                    # "VALID" or "SAME"
node.attrs["fused_activation_function"]  # "RELU", "RELU6", "NONE", etc.
node.attrs["dilation_h_factor"]          # int
node.attrs["dilation_w_factor"]          # int
```

### Softmax Options

```python
node.attrs["beta"]    # float (e.g. 1.0)
```

### Reshape Options

```python
node.attrs["new_shape"]    # list[int]
```

### Add / Arithmetic Options

```python
node.attrs["fused_activation_function"]   # "RELU", "RELU6", "NONE"
node.attrs["pot_scale_int16"]             # int
```

> Source: `test_tflite_backend.py::test_tflite_all_options_parsing`

---

## Pattern Matching on TFLite

```python
from neuron_toolkit.pattern import Pattern
from neuron_toolkit.backends.tflite.parser import TFLiteParser

parser = TFLiteParser("model.tflite")

pat = Pattern.op("ADD")
match = parser.pattern_detect(pat, start_node=parser.nodes[0])
match.start.op_type    # "ADD"
```

**Multi-backend pattern (works on both ONNX and TFLite):**

```python
from neuron_toolkit import NeuronGraph, Pattern
from neuron_toolkit.backends.tflite.parser import TFLiteParser, TFLiteNode

# Build a TFLite-backed NeuronGraph manually (e.g. for testing without a real file)
parser = TFLiteParser("dummy.tflite")
parser.nodes = [
    TFLiteNode("Relu", "r1", ["x"], ["y"]),
    TFLiteNode("Add",  "a1", ["y", "z"], ["out"]),
]
g = NeuronGraph(parser)

# Pattern: Add whose first input is Relu
relu_pat = Pattern.op("Relu").capture("relu_node")
add_pat  = Pattern.op("Add", relu_pat)

matches = g.findall(add_pat)
len(matches)                            # 1
matches[0].start.op_type               # "Add"
matches[0].bindings["relu_node"].op_type  # "Relu"
```

> Source: `test_multibackend.py::test_backend_agnostic_pattern_matching`

---

## TFLite Rewriting

### Replace a Node

```python
parser = TFLiteParser("model.tflite")
rw = parser.rewriter()
node = parser.nodes[0]   # e.g. ADD node

rw.replace([node], "MyFusion", node.input, node.output)
saved_bytes = rw.build("fused.tflite")   # saves to file AND returns bytes

new_parser = TFLiteParser("fused.tflite")
new_parser.nodes[0].op_type   # "MyFusion"
```

### Register a New Tensor (Constant)

```python
import numpy as np

data = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()

rw.register_tensor(
    "my_const",    # tensor name
    [3],           # shape
    "float32",     # dtype
    buffer_data=data,   # raw bytes (optional — for constant tensors)
)

# Use it in a replacement
rw.replace([node], "ADD", [node.input[0], "my_const"], node.output)
rw.build("output.tflite")

new_parser = TFLiteParser("output.tflite")
val = new_parser.tensor_map["my_const"]
val.shape    # (3,)
val.dtype    # float32
np.allclose(val, [1.0, 2.0, 3.0])   # True
```

> Source: `test_tflite_backend.py::test_tflite_rewriter_register_tensor`

---

### Topological Sort in TFLite Rewriter

When you stage edits out of order, the rewriter automatically topologically sorts the final node list:

```python
rw.replace([node], "ADD",       ["new_inter"], node.output, name="OpB")
rw.insert_before(node, "MyFusion", node.input, ["new_inter"], name="OpA")
rw.register_tensor("new_inter", [1, 10], "float32")
rw.build("toposort.tflite")

new_parser = TFLiteParser("toposort.tflite")
# OpA comes before OpB because OpB depends on OpA's output
new_parser.nodes[0].op_type   # "MyFusion"  (OpA)
new_parser.nodes[1].op_type   # "ADD"       (OpB)
```

> Source: `test_tflite_backend.py::test_tflite_rewriter_topological_sort`

---

### Write Op Options (e.g. RESHAPE with new_shape)

```python
rw.replace([node], "RESHAPE", node.input, node.output, new_shape=[2, 5])
rw.build("output.tflite")

new_parser = TFLiteParser("output.tflite")
new_parser.nodes[0].attrs.get("new_shape")   # [2, 5]
```

> Source: `test_tflite_backend.py::test_tflite_rewriter_options_serialization`

---

## Querying Parameters (Constant Inputs) on TFLite

```python
# After registering a constant tensor and building:
new_parser = TFLiteParser("query_params.tflite")
q = new_parser.find().op("ADD")

# has_params(): only nodes with constant (initializer) inputs
q.has_params().count()   # 1

# tensor(): returns dict of {const_tensor_name: np.ndarray}
params = q.tensor()
assert "my_const_tensor" in params
assert "input" not in params         # "input" is a graph input, not a constant

# single_tensor: shorthand when there's exactly one constant input
q.single_tensor.shape    # (3,)
```

> Source: `test_tflite_backend.py::test_tflite_query_params`

---

## Saving TFLite Models

```python
graph = NeuronGraph("model.tflite")
graph.save("copy.tflite")   # copies to new path

# From bytes source
graph = NeuronGraph(tflite_bytes)
graph.save("from_bytes.tflite")

# NOTE: Cannot save if loaded from a tflite.Model object
import tflite
model_obj = tflite.Model.GetRootAsModel(tflite_bytes, 0)
g = NeuronGraph(model_obj)
# g.save("x.tflite")  → raises ValueError: parsed object
```

> Source: `test_unified_graph_dispatch.py::test_save_tflite_*`

---

## What's next?

→ [10 — API Reference & Edge Cases](./10_api_reference.md) — complete API surface and gotchas.
