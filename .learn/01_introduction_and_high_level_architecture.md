# Module 01: Introduction & High-Level Architecture

Welcome to the `neuron-toolkit` masterclass! This module teaches you the core vision, architecture, and design decisions behind `neuron-toolkit` from first principles.

---

## 1. What is `neuron-toolkit`?

When working with deep learning models, engineers and researchers often need to inspect, query, modify, slice, or optimize model computational graphs.

Historically, this meant using framework-specific tools:

- **ONNX GraphSurgeon** or standard `onnx` APIs for ONNX models.
- **FlatBuffers / LiteRT / TFLite interpreter** for TFLite models.

`neuron-toolkit` is a **unified, multi-backend Python library** that bridges this gap. It provides a clean, abstract, framework-agnostic interface to:

1. **Load** both `.onnx` and `.tflite` model formats seamlessly into a unified graph object (`NeuronGraph`).
2. **Query** nodes using a fluent, method-chaining syntax (`NeuronQuery`).
3. **Match symbolic patterns** using a intuitive domain-specific language (`Pattern` DSL) and a robust graph search engine (`PatternDetector`).
4. **Rewrite graphs** safely using staged edits (`NeuronRewriter`).
5. **Slice subgraphs** automatically by tracing boundaries across directed node connections (`NeuronGraph.slice`).
6. **Cross-verify numerical output** against PyTorch modules layer-by-layer (`verify_against_pytorch`).

---

## 2. High-Level Architecture Overview

Here is how the modules inside `src/neuron_toolkit/` fit together:

```
                              [ Model Source ]
                      (.onnx file / .tflite flatbuffer)
                                     │
                                     ▼
                              ┌──────────────┐
                              │ NeuronGraph  │  <-- Unified Public Entry Point
                              └──────┬───────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
        ┌────────────────┐                      ┌──────────────────┐
        │   ONNXParser   │                      │   TFLiteParser   │
        └───────┬────────┘                      └────────┬─────────┘
                │                                        │
                └───────────────────┬────────────────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │   _GraphShim       │  <-- Generic Graph Interface
                         └──────────┬─────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
┌─────────────────┐       ┌───────────────────┐      ┌──────────────────┐
│   NeuronQuery   │       │  PatternDetector  │      │  NeuronRewriter  │
│ (Fluent Filter) │       │ (DFS Matcher Engine)│     │  (Staged Edits)  │
└─────────────────┘       └───────────────────┘      └──────────────────┘
```

---

## 3. Directory & File Structure Breakdown

The codebase is structured under `src/neuron_toolkit`:

| File / Folder      | Purpose                                                                                                                                                        |
| :----------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`      | Public package exports (`NeuronGraph`, `NeuronQuery`, `Pattern`, `NeuronRewriter`, etc.).                                                                      |
| `graph.py`         | Unified public interface (`NeuronGraph`) wrapping ONNX and TFLite backends, including verification & passes.                                                   |
| `_types.py`        | Shared type definitions like `TensorMap = dict[str, np.ndarray]`.                                                                                              |
| `_utils.py`        | Stand-in `_GraphShim` class and NetworkX boundary tracing (`trace_subgraph_boundaries`).                                                                       |
| `parser.py`        | Convenience re-exports for backend parsers and lazy tensor maps.                                                                                               |
| `rewriter.py`      | Public wrapper (`NeuronRewriter`) around backend-specific rewriters.                                                                                           |
| `query/core.py`    | `NeuronQuery` engine offering filtering, set ops (`union`, `intersect`), and graph traversal.                                                                  |
| `pattern/`         | Pattern DSL (`dsl.py`), activation constructors (`_activations.py`), DFS graph matcher (`_matcher.py`), detector (`detector.py`), match results (`models.py`). |
| `backends/base.py` | Abstract Base Classes `BaseParser` and `BaseRewriter`.                                                                                                         |
| `backends/onnx/`   | ONNX parser, staged rewriter, attribute extractors, and shape inference tools.                                                                                 |
| `backends/tflite/` | TFLite FlatBuffers parser, staged rewriter, FlatBuffers table cloners, and options serializers (`opt_*.py`).                                                   |

---

## 4. Fundamental Design Philosophy

### 1. Unified Interface over Direct Representation

Whether a model comes from PyTorch exported to ONNX or TensorFlow exported to TFLite FlatBuffers, `NeuronGraph` provides identical properties:

- `.nodes`: List of graph operators.
- `.tensor_map`: Map of parameter names to `numpy` arrays.
- `.shape_info`: Dictionary mapping tensor names to `(rank, dtype_str)`.

### 2. Lazy Evaluation & Deferred Execution

- Initializers/weight tensors are NOT loaded into memory all at once. They use `LazyTensorMap` to deserialize arrays only when explicitly accessed.
- Rewrites are NOT executed inline during replacement calls. Edits are staged inside `NeuronRewriter` and applied topologically in a single `build()` call.

### 3. Graph Theory First Principles

Graph operations like finding node ancestors, topological sorting, detecting cycles, and subgraph boundary extraction rely on **NetworkX directed graphs (`DiGraph`)**, treating operators as nodes and tensors as edges (or dual node types).

---

## 5. Quick Usage Example

Here is how simple it is to use `neuron-toolkit`:

```python
import numpy as np
from neuron_toolkit import NeuronGraph, Pattern

# 1. Load an ONNX or TFLite model automatically
graph = NeuronGraph.load("resnet18.onnx")

# 2. Query nodes using the fluent API
convs = graph.query().op("Conv").attr("group", 1)
print(f"Found {convs.count()} standard convolution layers.")

# 3. Define a symbolic pattern to find Conv + Relu subgraphs
pattern = Pattern.op("Relu", Pattern.op("Conv").capture("conv"))

# 4. Find all matches
matches = graph.findall(pattern)
for match in matches:
    print(f"Matched Conv node: {match.bindings['conv'].name}")

# 5. Execute an optimization pass (e.g., Conv + BN fusion)
rewriter = graph.passes.fuse_conv_bn()
fused_model = rewriter.build("resnet18_fused.onnx")
```

In the next module, we will explore the **core data structures and types** that power this library.
