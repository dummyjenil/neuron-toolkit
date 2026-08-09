# 07 — Graph Slicing

**Graph Slicing** extracts a contiguous sub-graph from a model, given boundary tensors.  
The result is a new `NeuronGraph` you can inspect, save, or use for comparison.

---

## `graph.slice(start, end)` — Extract Between Two Tensors

```python
from neuron_toolkit import NeuronGraph

graph = NeuronGraph("model.onnx")
# Model: Identity → Relu → Add(B) → Mul(C) → Sigmoid and Tanh (6 nodes total)

# Slice between "relu_out" (input boundary) and "mul_out" (output boundary)
sliced = graph.slice("relu_out", "mul_out")

len(sliced.nodes)                    # 2 (only Add and Mul are kept)
{n.op_type for n in sliced.nodes}   # {"Add", "Mul"}

# The boundary tensors become the graph's new I/O
sliced._backend.graph_inputs    # contains "relu_out"
sliced._backend.graph_outputs   # contains "mul_out"
```

### Save and Reload

```python
sliced.save("sliced.onnx")

reloaded = NeuronGraph("sliced.onnx")
len(reloaded.nodes)   # 2
```

> Source: `test_slicing.py::test_onnx_slicing`

---

## `graph.slice_node(node)` — Extract a Single Node

Extracts exactly one node into its own standalone graph:

```python
# By node object
relu_node = [n for n in graph.nodes if n.op_type == "Relu"][0]
sliced = graph.slice_node(relu_node)

len(sliced.nodes)            # 1
sliced.nodes[0].op_type     # "Relu"

sliced.save("relu_only.onnx")

# By name string
sliced = graph.slice_node("node_id")
sliced.nodes[0].op_type     # "Identity"
```

> Source: `test_slicing.py::test_onnx_single_node_extraction`, `test_unified_graph_dispatch.py::test_slice_and_slice_node`

---

## `graph.slice()` with Node Objects as Endpoints

You can also pass node objects (not just tensor name strings):

```python
add_node = graph.query().op("Add").single_node

sliced = graph.slice(add_node, add_node)
len(sliced.nodes)             # 1
sliced.nodes[0].op_type       # "Add"
```

> Source: `test_unified_graph_dispatch.py::test_slice_and_slice_node`

---

## Error Handling

```python
# Invalid tensor names raise ValueError
graph.slice("not_a_tensor", "also_not_a_tensor")
# ValueError: start or end ...
```

> Source: `test_unified_graph_dispatch.py::test_slice_invalid_endpoints_raise`

---

## TFLite Slicing

Slicing works identically on TFLite models:

```python
# TFLite model: input → ABS → abs_out → ADD(const_w) → add_out → SUB → output
graph = NeuronGraph("model.tflite")
assert len(graph.nodes) == 3

# Slice between abs_out and add_out — should keep only ADD and its constant
sliced = graph.slice("abs_out", "add_out")

len(sliced.nodes)              # 1
sliced.nodes[0].op_type        # "ADD"

sliced._backend.graph_inputs   # {"abs_out"}
sliced._backend.graph_outputs  # {"add_out"}
"const_w" in sliced.tensor_map # True (the constant weight is preserved)

sliced.save("tflite_sliced.tflite")

reloaded = NeuronGraph("tflite_sliced.tflite")
len(reloaded.nodes)            # 1
reloaded.nodes[0].op_type      # "ADD"
"const_w" in reloaded.tensor_map  # True
```

```python
# Single node extraction — TFLite
abs_node = graph.nodes[0]
sliced = graph.slice_node(abs_node)

sliced.nodes[0].op_type   # "ABS"
sliced.save("abs_only.tflite")
```

> Source: `test_slicing.py::test_tflite_slicing`, `test_tflite_single_node_extraction`

---

## Full ONNX Slicing Example

```python
from neuron_toolkit import NeuronGraph

graph = NeuronGraph("complex.onnx")
# 6 nodes: Identity → Relu → Add(B) → Mul(C) → Sigmoid & Tanh

# Extract the "middle" of the graph (Add and Mul only)
sliced = graph.slice("relu_out", "mul_out")

print(f"Nodes in slice: {len(sliced.nodes)}")         # 2
print({n.op_type for n in sliced.nodes})               # {'Add', 'Mul'}
print(sliced._backend.graph_inputs)                    # {'relu_out'}
print(sliced._backend.graph_outputs)                   # {'mul_out'}

sliced.save("middle_slice.onnx")

# Verify it reloads correctly
reloaded = NeuronGraph("middle_slice.onnx")
assert len(reloaded.nodes) == 2
```

---

## What's next?

→ [08 — PyTorch Comparison](./08_pytorch_comparison.md) — verify a slice numerically against a PyTorch reference.
