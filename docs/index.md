# Neuron-Toolkit Documentation

**Neuron-Toolkit** is a unified Python library designed for multi-backend neural network graph parsing, querying, pattern-matching, rewriting, slicing, and verification across **ONNX** and **TensorFlow Lite (TFLite)** models.

---

## Key Features

- **Unified Graph Abstraction (`NeuronGraph`)**: Load ONNX models (`.onnx`, `ModelProto`) and TFLite flatbuffers (`.tflite`, `.bin`, bytes, `tflite.Model`) under a single API.
- **Fluent Query Engine (`NeuronQuery`)**: Inspect, filter, and traverse model nodes using chainable query methods (filtering by op type, name, attributes, output rank, data type, input/output tensors, set operations, and topological sorting).
- **Symbolic Pattern Matching DSL (`Pattern`)**: Express complex subgraphs using operator wildcards, constants, value constraints, attribute constraints, arithmetic operator overloads (`+`, `-`, `*`, `/`, `**`), and 20+ built-in activation pattern constructors (ReLU, GeLU, Swish, Mish, Softmax, etc.).
- **Smart Backtracking & Commutative Matcher (`PatternDetector`)**: Automatically handles commutative operations (such as `Add` and `Mul`) via input permutation matching, with referential consistency and subgraph boundary bounds.
- **Staged Graph Rewriter (`NeuronRewriter`)**: Safely queue and execute node replacements (`replace`, `replace_from_result`), deletions (`delete`), node insertions (`insert_before`), and new weight tensor registrations (`register_initializer`) without invalidating graph traversal state.
- **Subgraph Slicing & Boundary Extraction (`slice`, `slice_node`)**: Extract any sub-segment of an ONNX or TFLite model into a standalone, runnable model file using NetworkX directed reachability tracing.
- **PyTorch Cross-Verification (`compare_outputs`, `verify_against_pytorch`)**: Automatically execute and compare sliced subgraphs against PyTorch module layers using `onnxruntime` or `ai_edge_litert`, calculating MAE, MSE, Cosine Similarity, and shape matching.
- **Graph Optimization Passes (`GraphPasses`)**: Built-in pattern-based graph transformations, such as `fuse_conv_bn` (Conv + BatchNormalization folding).
- **Quantization & Sparsity Metadata Inspection**: Extract TFLite per-tensor and per-channel quantization parameters (scales, zero points, quantized dimensions) and sparsity maps effortlessly.

---

## Quickstart Guide

### 1. Loading a Model

```python
from neuron_toolkit import NeuronGraph

# Load ONNX model
graph_onnx = NeuronGraph.load("model.onnx")

# Load TFLite model
graph_tflite = NeuronGraph.load("model.tflite")

# Inspect graph statistics
print(f"Nodes count: {len(graph_onnx.nodes)}")
print(f"Weight tensors: {list(graph_onnx.tensor_map.keys())}")
```

### 2. Querying Graph Nodes

```python
# Find all Conv nodes with output rank 4
conv_nodes = graph_onnx.query().op("Conv").rank(4)

# Get direct children of entry nodes
entry_successors = graph_onnx.query().entry_nodes.children()

# Filter nodes possessing weight tensors
weighted_nodes = graph_onnx.query().has_params()
```

### 3. Subgraph Pattern Matching

```python
from neuron_toolkit import Pattern

# Define a Swish activation pattern: x * Sigmoid(x)
x = Pattern.any()
swish_pattern = x * Pattern.sigmoid(x)

# Match pattern in graph
matches = graph_onnx.findall(swish_pattern)
print(f"Found {len(matches)} Swish patterns!")
```

### 4. Graph Rewriting & Transformation

```python
# Fuse Conv + BatchNormalization
rewriter = graph_onnx.passes.fuse_conv_bn()

# Rebuild model with changes applied and save to disk
fused_model = rewriter.build(output_path="fused_model.onnx")
```

### 5. PyTorch Verification

```python
import torch

# Verify sliced model outputs against a PyTorch module
@graph_onnx.verify_against_pytorch(start_points=["input_node"], end_points=["output_node"])
def run_pytorch_layer(x):
    return torch.relu(x)

# Invoking wrapper automatically calculates numerical similarity metrics
out = run_pytorch_layer(torch.randn(1, 3, 224, 224))
print(run_pytorch_layer.similarity_report)
```

---

## Documentation Modules

- [NeuronGraph API Reference](neuron-graph.md)
- [Query Engine Guide](query-engine.md)
- [Pattern Matching DSL Reference](pattern-matching.md)
- [Graph Rewriting Guide](graph-rewriting.md)
- [Backend Deep Dive (ONNX & TFLite)](backends.md)
- [Model Slicing & PyTorch Verification](slicing-and-verification.md)
- [Graph Optimization Passes](optimization-passes.md)
