# Model Slicing & PyTorch Cross-Verification

**Neuron-Toolkit** includes graph slicing and numerical cross-verification against PyTorch layers and functions.

---

## Subgraph Boundary Extraction & Model Slicing

Graph slicing isolates a target segment of a neural network model between designated starting points (`start_points`) and ending points (`end_points`).

### `trace_subgraph_boundaries` Function

Located in `neuron_toolkit._utils`:

```python
def trace_subgraph_boundaries(
    nodes: Sequence[object],
    tensor_map: Mapping[str, object],
    start_points: list[str | object] | str | object,
    end_points: list[str | object] | str | object,
    original_outputs: set[str] | None = None,
) -> tuple[list[object], list[str], list[str]]: ...
```

Uses NetworkX directed reachability (`descendants` of start points intersected with `ancestors` of end points) to trace:

1. **`kept_ops`**: All operator nodes lying on any path from `start_points` to `end_points`.
2. **`new_inputs`**: Boundary input tensor names (tensors consumed within the subgraph that are not produced inside it and are not initializers/weights).
3. **`new_outputs`**: Boundary output tensor names (explicit end point tensors, or internal tensors consumed outside the subgraph).

---

## `NeuronGraph.slice` Methods

### 1. `graph.slice`

```python
sliced_graph = graph.slice(
    start_points: list[str | object] | str | object,
    end_points: list[str | object] | str | object,
    output_path: str | None = None
) -> NeuronGraph
```

Extracts the sliced subgraph into a new, runnable `NeuronGraph`. If `output_path` is specified, the sliced model file is written to disk (.onnx or .tflite).

### 2. `graph.slice_node`

```python
single_node_graph = graph.slice_node(node, output_path=None) -> NeuronGraph
```

Shortcut to extract a single operator node into a standalone model.

---

## PyTorch Cross-Verification API

### 1. Direct Comparison: `graph.compare_outputs`

```python
report = graph.compare_outputs(
    start_points: list[str | object] | str | object,
    end_points: list[str | object] | str | object,
    pytorch_module: Any,
    inputs_dict: dict[str, Any],
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> dict[str, Any]
```

#### Execution Steps

1. Executes PyTorch module/function on `inputs_dict` tensors under `torch.no_grad()`.
2. Slices the model using `graph.slice(start_points, end_points)`.
3. Runs the sliced model using `onnxruntime` (for ONNX) or `ai_edge_litert.interpreter` (for TFLite).
4. Matches model outputs against PyTorch outputs and calculates numerical similarity metrics.

#### Returned Metric Report Schema

```python
{
    "output_tensor_name": {
        "mae": float,                # Mean Absolute Error
        "mse": float,                # Mean Squared Error
        "cosine_similarity": float, # Cosine similarity (1.0 = identical direction)
        "all_close": bool,           # np.allclose(rtol=rtol, atol=atol)
        "shape_match": bool          # True if output tensor shapes match exactly
    }
}
```

---

### 2. Decorator Verification: `@graph.verify_against_pytorch`

```python
@graph.verify_against_pytorch(
    start_points=["input_tensor"],
    end_points=["output_tensor"],
    rtol=1e-5,
    atol=1e-8
)
def my_pytorch_layer(x):
    return torch.nn.functional.relu(x)

# Executing decorated function automatically runs verification in the background
output = my_pytorch_layer(input_tensor)

# Access similarity report on function wrapper
print(my_pytorch_layer.similarity_report)
```
