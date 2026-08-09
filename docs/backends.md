# Backends: ONNX & TFLite Support

**Neuron-Toolkit** features backend parsers and rewriters for both **ONNX** and **TensorFlow Lite (TFLite)** models under the `BaseParser` and `BaseRewriter` abstract base classes.

---

## Backend Base Interfaces (`BaseParser` & `BaseRewriter`)

### `BaseParser` Abstract Interface

- **`find() -> NeuronQuery`**: Returns a query object over model nodes.
- **`pattern_detect(pattern, start_node, end_node) -> MatchResult | None`**: Finds a pattern match.
- **`rewriter() -> BaseRewriter`**: Instantiates a backend rewriter.
- **`get_node_attrs(node) -> dict[str, object]`**: Extracts backend node attributes.
- **`is_constant_node(node) -> bool`**: Checks if node is constant-provider.
- **`get_constant_value(node) -> object | None`**: Extracts constant value array.
- **`summary() -> str`**: Generates human-readable graph summary.

---

## ONNX Backend (`ONNXParser` & `ONNXRewriter`)

### `ONNXParser`

Located in `neuron_toolkit.backends.onnx.parser`.

- **Input formats**: File path (`.onnx`), raw serialized string/bytes, or `onnx.ModelProto`.
- **Automatic Shape Inference**: Runs `onnx.shape_inference.infer_shapes()` upon loading by default (`infer_shapes=True`).
- **`LazyTensorMap`**: Lazily converts ONNX initializer `TensorProto` data into NumPy arrays on access.
- **Node attributes**: Extracted via `_node_attrs()` in `backends.onnx.utils`, mapping ONNX attribute types (`INT`, `FLOAT`, `INTS`, `FLOATS`, `TENSOR`, `STRING`) to Python primitives and NumPy arrays.
- **Model Slicing**: Wraps `onnx.utils.extract_model` to extract sub-models.

---

## TFLite Backend (`TFLiteParser` & `TFLiteRewriter`)

### `TFLiteParser`

Located in `neuron_toolkit.backends.tflite.parser`.

- **Input formats**: File path (`.tflite`, `.bin`), raw bytes (Flatbuffer), or pre-parsed `tflite.Model`.
- **`LazyTensorMap`**: Extracts flatbuffer buffer data directly into NumPy arrays lazily based on buffer index mapping.
- **Unified Node Schema (`TFLiteNode`)**: Maps TFLite Flatbuffer operators into standard node representations:
  - `op_type`: Built-in operator name (e.g. `"CONV_2D"`, `"ADD"`) or custom operator string.
  - `name`: Generated unique node name (e.g. `"CONV_2D_0"`).
  - `input`: List of input tensor names.
  - `output`: List of output tensor names.
  - `attrs`: Extracted TFLite operator options dictionary.

### Quantization & Sparsity Information Extraction

`TFLiteParser` extracts quantization parameters and sparsity metadata for every tensor into dedicated dictionaries:

#### `quantization_info` Metadata Schema

```python
parser = graph._backend  # TFLiteParser instance
quant_meta = parser.quantization_info["tensor_name"]
# Returns dict:
# {
#     "min": [...],
#     "max": [...],
#     "scale": [float, ...],
#     "zero_point": [int, ...],
#     "quantized_dimension": int
# }
```

#### `sparsity_info` Metadata Schema

```python
sparsity_meta = parser.sparsity_info["tensor_name"]
# Returns dict containing dim_metadata, traversal_order, etc.
```

---

## TFLite Option Parsers

`neuron_toolkit.backends.tflite` contains option parsers that decode Flatbuffer options tables into standard Python dictionaries:

- **`opt_activation`**: `ConcatEmbeddingsOptions`, `L2NormOptions`, `LocalResponseNormalizationOptions`, `RNNOptions`, `SequenceRNNOptions`
- **`opt_arithmetic`**: `ArgMaxOptions`, `ArgMinOptions`, `AddOptions`, `SubOptions`, `MulOptions`, `DivOptions`
- **`opt_control_flow`**: `IfOptions`, `WhileOptions`, `CallOnceOptions`
- **`opt_misc`**: `AssignVariableOptions`, `ReadVariableOptions`, `CustomQuantizationOptions`, `StableHLOCompositeOptions`
- **`opt_nn`**: `Conv2DOptions`, `DepthwiseConv2DOptions`, `TransposeConvOptions`, `FullyConnectedOptions`, `SoftmaxOptions`, `Pool2DOptions`
- **`opt_reduction`**: `ReducerOptions`
- **`opt_stablehlo`**: `ReduceWindowOptions`, `StablehloBroadcastInDimOptions`, `StablehloCompareOptions`, `StablehloConcatenateOptions`, `StablehloDotGeneralOptions`, `StablehloReduceOptions`, `StablehloSliceOptions`
- **`opt_tensor`**: `ReshapeOptions`, `TransposeOptions`, `SliceOptions`, `PackOptions`, `UnpackOptions`, `GatherOptions`, `ScatterNdOptions`, `ShapeOptions`, `StridedSliceOptions`
