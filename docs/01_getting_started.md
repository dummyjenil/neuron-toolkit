# 01 — Getting Started

## What is `NeuronGraph`?

`NeuronGraph` is the **main entry point** of neuron-toolkit. You pass it a model (ONNX or TFLite) and it gives you a unified object for inspection, querying, pattern-matching, and rewriting — regardless of the backend format.

---

## Loading a Model

### From a file path (auto-detects format by extension)

```python
from neuron_toolkit import NeuronGraph

# ONNX  — any path ending in .onnx
graph = NeuronGraph("model.onnx")

# TFLite — any path ending in .tflite or .bin
graph = NeuronGraph("model.tflite")
```

> Source: `test_unified_graph_dispatch.py::test_graph_from_str_dispatches_by_extension`

---

### From bytes (auto-detects via TFL3 magic bytes)

```python
# ONNX bytes
data = open("model.onnx", "rb").read()
graph = NeuronGraph(data)

# TFLite bytes — must have the real b"TFL3" file identifier
tflite_data = open("model.tflite", "rb").read()
graph = NeuronGraph(tflite_data)
```

> Source: `test_unified_graph_dispatch.py::test_graph_from_onnx_bytes`, `test_graph_from_tflite_bytes`

---

### From an ONNX ModelProto object

```python
import onnx

model_proto = onnx.load("model.onnx")
graph = NeuronGraph(model_proto)
```

> Source: `test_unified_graph_dispatch.py::test_graph_from_onnx_model_proto`

---

### From a low-level parser

```python
from neuron_toolkit.backends.onnx.parser import ONNXParser
from neuron_toolkit.backends.tflite.parser import TFLiteParser

graph = NeuronGraph(ONNXParser("model.onnx"))
graph = NeuronGraph(TFLiteParser("model.tflite"))
```

> Source: `test_unified_graph_dispatch.py::test_graph_from_onnx_parser`

---

### Using `NeuronGraph.load()` (explicit static constructor)

```python
graph = NeuronGraph.load("model.onnx")
graph = NeuronGraph.load("model.tflite")

# Raises ValueError for unsupported extensions:
# NeuronGraph.load("model.ir")  →  ValueError: Unsupported model format
```

> Source: `test_unified_graph_dispatch.py::test_load_supported_extensions`, `test_load_unsupported_extension`

---

## Inspecting Basic Properties

```python
graph = NeuronGraph("model.onnx")

# --- Nodes ---
print(len(graph.nodes))        # total operator count
print(graph.nodes[0].op_type)  # e.g. "Identity", "Conv", "Relu"
print(graph.nodes[0].name)     # node name string

# --- Initializers / Weight tensors ---
# tensor_map is a lazy dict: keys are tensor names, values are np.ndarray
print(set(graph.tensor_map))           # {'B', 'C'}
b_array = graph.tensor_map["B"]       # np.ndarray, loaded on first access
print(b_array.shape)                   # (1, 3)

# --- Shape information ---
# shape_info[tensor_name] → (rank: int | None, dtype: str)
rank, dtype = graph.shape_info["output"]
print(rank)   # 2
print(dtype)  # "float32"

# --- Graph inputs/outputs ---
# Accessible via the internal backend parser
print(graph._backend.graph_inputs)    # set of input tensor names
print(graph._backend.graph_outputs)   # set of output tensor names
```

> Source: `comprehensive_test.py::test_neuron_graph_basics`, `test_unified_graph_dispatch.py::test_graph_properties`

---

## Getting All Weights as a Dict

```python
weights = graph.get_weights()
# Returns {name: np.ndarray} for all initializer tensors

print(set(weights))          # {'B', 'C'}
print(type(weights["B"]))    # <class 'numpy.ndarray'>
```

> Source: `test_unified_graph_dispatch.py::test_graph_get_weights`

---

## Saving a Model

```python
graph.save("output.onnx")    # saves ONNX to file
graph.save("output.tflite")  # saves TFLite to file
```

> **Note:** You cannot save a TFLite model that was loaded from a raw `tflite.Model` object (parsed in memory). Load from path or bytes instead.

> Source: `test_unified_graph_dispatch.py::test_save_onnx`, `test_save_tflite_bytes_source`

---

## Using the Low-Level ONNX Parser Directly

For advanced use cases, you can bypass `NeuronGraph` and use `ONNXParser` directly:

```python
from neuron_toolkit.parser import ONNXParser

parser = ONNXParser("model.onnx")

print(len(parser.nodes))          # 3
print("A" in parser.graph_inputs) # True
print("output" in parser.graph_outputs)  # True
print("B" in parser.tensor_map)   # True

# Print a human-readable summary
print(parser.summary())
# Output example:
#   Nodes        : 3
#   Identity, Add, Mul
#   ...
```

> Source: `test_parser.py`, `test_parser_deep.py`

---

## `tensor_map` — Lazy Loading

The `tensor_map` is a `LazyTensorMap` — it **does not load any data** until you actually access a key:

```python
from neuron_toolkit.backends.onnx.parser import ONNXParser, LazyTensorMap

parser = ONNXParser("model.onnx")
tm = parser.tensor_map

assert isinstance(tm, LazyTensorMap)
assert tm._cache == {}         # nothing loaded yet

b = tm["B"]                    # first access → parses and caches
assert "B" in tm._cache        # now cached
assert tm["B"] is b            # same object returned (no re-parse)

# Full mapping protocol supported:
len(tm)                        # number of initializers
list(tm.keys())
list(tm.values())
dict(tm.items())

tm.get("missing")              # returns None
tm.get("missing", "default")   # returns "default"
# tm["missing"]                → raises KeyError
```

> Source: `test_onnx_tensor_map.py`

---

## Error Handling

```python
# Non-existent file
ONNXParser("non_existent.onnx")     # raises FileNotFoundError / onnx error

# Corrupted file
ONNXParser("corrupt.onnx")          # raises Exception

# TFLite bytes without TFL3 magic treated as ONNX and fails
NeuronGraph(corrupted_tflite_bytes)  # raises Exception

# Unsupported format in .load()
NeuronGraph.load("/tmp/model.ir")   # raises ValueError: Unsupported model format
```

> Source: `test_parser_deep.py::test_parser_invalid_path`, `test_parser_corrupt_model`

---

## What's next?

→ [02 — Query Engine](./02_query_engine.md) — filter and traverse nodes fluently.
