# Staged Graph Rewriting Engine (`NeuronRewriter`)

`NeuronRewriter` provides a staged graph modification engine for **ONNX** and **TensorFlow Lite (TFLite)** models. Edits (node replacements, insertions, deletions, initializer updates) are staged in memory without invalidating node iterators or graph indices during traversal, and applied upon calling `.build()`.

---

## Obtaining a Rewriter

```python
from neuron_toolkit import NeuronGraph

graph = NeuronGraph.load("model.onnx")
rewriter = graph.rewriter()
```

---

## Staged Edit Methods

### 1. Node Replacement

```python
rewriter.replace(
    nodes: Sequence[object],
    new_op: str,
    inputs: list[str],
    outputs: list[str],
    name: str | None = None,
    **attrs: object
) -> NeuronRewriter
```

Schedules the removal of `nodes` and insertion of a new operator node `new_op` with specified `inputs`, `outputs`, and node attributes `attrs`.

---

### 2. Matched Subgraph Replacement (`replace_from_result`)

```python
rewriter.replace_from_result(
    result: MatchResult,
    new_op: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    name: str | None = None,
    **attrs: object
) -> NeuronRewriter
```

Convenience wrapper to replace an entire matched subgraph represented by a `MatchResult`. If `inputs` or `outputs` are omitted, boundary inputs and outputs are resolved automatically.

---

### 3. Node Deletion (`delete`)

```python
rewriter.delete(nodes: Sequence[object]) -> NeuronRewriter
```

Schedules the removal of `nodes` from the model graph.

---

### 4. Node Insertion (`insert_before`)

```python
rewriter.insert_before(
    target_node: object,
    new_op: str,
    inputs: list[str],
    outputs: list[str],
    name: str | None = None,
    **attrs: object
) -> NeuronRewriter
```

Inserts a new operator node `new_op` immediately upstream of `target_node`.

---

### 5. Constant Initializer Registration (`register_initializer`)

```python
rewriter.register_initializer(name: str, value: np.ndarray) -> NeuronRewriter
```

Registers a new NumPy array `value` as a named constant/initializer tensor in the graph. Essential for graph optimization passes (e.g. folded Conv + BatchNorm weights and bias).

---

### 6. Resetting Edits (`reset`)

```python
rewriter.reset() -> NeuronRewriter
```

Discards all staged modifications, clearing pending deletions, insertions, and initializers.

---

### 7. Building Modified Models (`build`)

```python
rewriter.build(output_path: str | None = None) -> object
```

Executes all pending edits, rebuilds the internal backend model structure, saves to `output_path` (if specified), and returns the modified model object (`onnx.ModelProto` or TFLite model bytes).

---

## Rewriting Example

```python
import numpy as np
from neuron_toolkit import NeuronGraph, Pattern

graph = NeuronGraph.load("model.onnx")

# Match Conv + Relu
conv = Pattern.op("Conv").capture("conv")
relu = Pattern.op("Relu", conv)

matches = graph.findall(relu)
rewriter = graph.rewriter()

for m in matches:
    conv_node = m.bindings["conv"]
    # Register new fused weight or attribute if needed
    rewriter.replace_from_result(m, "FusedConvRelu", **conv_node.attrs)

# Build transformed model
transformed_model = rewriter.build(output_path="fused_model.onnx")
```
