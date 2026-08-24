# `NeuronGraph` - Unified Graph Interface

The `NeuronGraph` class is the central entry point in **Neuron-Toolkit**. It wraps both **ONNX** and **TensorFlow Lite (TFLite)** backends, providing a standardized interface for loading, querying, pattern matching, rewriting, slicing, and exporting neural network model graphs.

---

## Class Signature

```python
class NeuronGraph:
    def __init__(self, source: str | bytes | object, **kwargs: Any) -> None: ...
```

### Initialization Modes

`NeuronGraph` supports flexible model sources:

1. **File Path (`str`)**: Auto-detects backend format based on extension (`.onnx`, `.tflite`, `.bin`).
2. **Raw Bytes (`bytes`)**: Checks Flatbuffer magic bytes at offset 4 (`b"TFL3"`) for TFLite; falls back to ONNX parser.
3. **ONNX `ModelProto`**: Direct wrap of `onnx.ModelProto` instances.
4. **TFLite `Model`**: Direct wrap of pre-parsed `tflite.Model` flatbuffer objects.
5. **Backend Parsers**: Accepts `ONNXParser` or `TFLiteParser` instances.

---

## Class Methods & Properties

### Constructor Methods

#### `NeuronGraph.load(path: str, **kwargs: Any) -> NeuronGraph`

Class method to load a model from a file path.

```python
graph = NeuronGraph.load("model.onnx")
# or
graph = NeuronGraph.load("model.tflite")
```

---

### Core Graph Properties

#### `.nodes -> list[object]`

Returns the list of graph operator nodes (`NodeProto` for ONNX, `TFLiteNode` for TFLite).

#### `.tensor_map -> dict[str, np.ndarray]`

Returns a dictionary mapping initializer/constant parameter tensor names to NumPy arrays (lazy-loaded via `LazyTensorMap`).

#### `.shape_info -> dict[str, tuple[int | None, str | None]]`

Returns a mapping from tensor names to output shape/data type info tuples: `(rank, dtype_str)`.

#### `.passes -> GraphPasses`

Returns the `GraphPasses` namespace for executing graph optimization passes.

---

### High-Level API Methods

#### `.query() -> NeuronQuery`

Returns a `NeuronQuery` object over all graph nodes.

#### `.match(pattern: Pattern) -> MatchResult | None`

Finds the first match of the specified symbolic pattern in the model.

#### `.findall(pattern: Pattern) -> list[MatchResult]`

Finds all occurrences of the pattern in the model graph.

#### `.replace(pattern: Pattern, new_op: str, name: str | None = None, **attrs: object) -> NeuronRewriter`

Shortcut method to find all pattern occurrences and stage their replacement with a new operator `new_op`.

#### `.rewriter() -> NeuronRewriter`

Returns a `NeuronRewriter` bound to this graph for staged graph transformations.

#### `.slice(start_points, end_points, output_path=None) -> NeuronGraph`

Slices the model graph between specified start and end points and returns a new `NeuronGraph`.

#### `.slice_node(node, output_path=None) -> NeuronGraph`

Extracts a single node as a standalone model graph.

#### `.get_weights() -> dict[str, Any]`

Returns a dictionary containing all weight/initializer parameter arrays in the graph.

#### `.save(path: str) -> None`

Saves the model to the designated path on disk.

```python
graph.save("optimized_model.onnx")
```

#### `.to_dict() -> dict[str, Any]`

Exports the complete model graph structure into a Python dictionary without raw weight buffers, replacing them with detailed statistical summaries (sum, mean, min, max, std, sparsity).

#### `.to_json(path: str | None = None, indent: int = 2) -> str | None`

Exports the model graph as a JSON string or writes directly to a JSON file. Ideal for Netron-like visualizers and lightweight graph analysis.

```python
# Export to JSON string
json_str = graph.to_json()

# Export directly to file
graph.to_json("model_graph.json")
```

#### `.export_graph_json(path=None, indent=2)` / `.to_graph_json(path=None, indent=2)`

Aliases for `.to_json()`.

#### `.save_safetensors(path: str, metadata: dict[str, str] | None = None) -> None`

Saves all model weight and bias parameters into a standard `.safetensors` file. Compatible with Hugging Face Safetensors.

```python
# Save weights and biases into a safetensors file
graph.save_safetensors("model_weights.safetensors")
```

#### `.export_safetensors(path: str, metadata: dict[str, str] | None = None) -> None`

Alias for `.save_safetensors()`.

#### `NeuronGraph.from_json(json_source, weights=None, seed=None) -> NeuronGraph`

Reconstructs a complete model graph from a JSON graph file, string, or dictionary, and optional weights:

- **JSON + Safetensors (`weights="model.safetensors"`)**: Rebuilds the exact graph topology with original weights and biases.
- **JSON only (`weights=None`)**: Rebuilds the graph topology with randomly initialized weights/biases matching the shapes and dtypes in the JSON specification.

```python
# Load with exact weights from safetensors
rebuilt_graph = NeuronGraph.from_json("model_graph.json", weights="model_weights.safetensors")

# Load with random initialization
random_graph = NeuronGraph.from_json("model_graph.json", seed=42)
```

#### `.load_safetensors(path: str, strict: bool = False) -> dict[str, list[str]]`

Loads weights from a `.safetensors` file and replaces the model initializers/biases in-place (similar to PyTorch's `load_state_dict`).

```python
# Replace weights in-place
status = graph.load_safetensors("new_weights.safetensors")
print(status["missing_keys"], status["unexpected_keys"])
```

#### `.replace_weights(weights_dict: dict[str, np.ndarray], strict: bool = False) -> dict[str, list[str]]`

Replaces weights and biases in-place from a NumPy array dictionary.

---

### Verification API Methods

#### `.compare_outputs(start_points, end_points, pytorch_module, inputs_dict, rtol=1e-5, atol=1e-8) -> dict[str, Any]`

Executes a sliced subgraph and PyTorch module on identical input data and produces comparative accuracy metrics (MAE, MSE, Cosine Similarity, All-close flag).

#### `.verify_against_pytorch(start_points, end_points, rtol=1e-5, atol=1e-8)`

Decorator to automatically verify PyTorch function or layer outputs against the model's sliced subgraph upon invocation.
