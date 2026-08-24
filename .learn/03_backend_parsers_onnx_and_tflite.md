# Module 03: Backend Parsers (ONNX & TFLite)

This module explains how model parsing works under the hood for both ONNX models and TFLite FlatBuffers.

---

## 1. Abstract Base Class `BaseParser` (`src/neuron_toolkit/backends/base.py`)

All model parsers inherit from `BaseParser`, defining a contract that every backend MUST satisfy:

```python
class BaseParser(ABC):
    @abstractmethod
    def find(self) -> NeuronQuery: ...

    @abstractmethod
    def pattern_detect(self, pattern: Pattern, ...) -> MatchResult | None: ...

    @abstractmethod
    def rewriter(self) -> BaseRewriter: ...

    @abstractmethod
    def get_node_attrs(self, node: object) -> dict[str, object]: ...

    @abstractmethod
    def is_constant_node(self, node: object) -> bool: ...

    @abstractmethod
    def get_constant_value(self, node: object) -> object | None: ...

    @abstractmethod
    def summary(self) -> str: ...
```

---

## 2. ONNX Parser (`src/neuron_toolkit/backends/onnx/parser.py`)

### Loading & Shape Inference

When an ONNX file or string is loaded:

```python
if isinstance(onnx_source, ModelProto):
    self.model = onnx_source
elif isinstance(onnx_source, bytes):
    self.model = onnx.load_model_from_string(onnx_source)
else:
    self.model = onnx.load(onnx_source)

if infer_shapes:
    self.model = onnx.shape_inference.infer_shapes(self.model)
```

1. **Shape Inference**: Calls `onnx.shape_inference.infer_shapes()` to automatically populate tensor rank and dtype annotations for every value info wire in the graph.
2. **`LazyTensorMap`**: Initializers are wrapped inside `LazyTensorMap` so arrays are decoded via `onnx.numpy_helper.to_array()` only when requested by key.

### ONNX `LazyTensorMap` Implementation:

```python
class LazyTensorMap(dict[str, "np.ndarray"]):
    def __init__(self, initializers: Sequence[onnx.TensorProto]) -> None:
        super().__init__()
        self._initializers = {t.name: t for t in initializers}
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._cache:
            return self._cache[key]
        if key in self._initializers:
            arr = numpy_helper.to_array(self._initializers[key])
            self._cache[key] = arr
            return arr
        return super().__getitem__(key)
```

### Attribute Extraction (`src/neuron_toolkit/backends/onnx/utils.py`)

ONNX node attributes use typed protocol buffers (`AttributeProto`). `_node_attrs()` converts them into standard Python objects:

- `FLOAT` -> `float`
- `INT` -> `int`
- `STRING` -> `str` (decoded UTF-8)
- `TENSOR` -> `numpy.ndarray`
- `INTS` -> `list[int]`

---

## 3. TFLite Parser (`src/neuron_toolkit/backends/tflite/parser.py`)

TFLite models are stored as Google FlatBuffer binary files (`.tflite`).

### Parsing FlatBuffers

The parser uses the `tflite` FlatBuffer library:

```python
import tflite

model = tflite.Model.GetRootAsModel(buffer_bytes, 0)
subgraph = model.Subgraphs(0) # Primary subgraph
```

### Building Nodes & Resolving OpCodes

TFLite operators store opcodes as integer indices referencing a global `OperatorCodes` table in the FlatBuffer:

```python
for i in range(subgraph.OperatorsLength()):
    op = subgraph.Operators(i)
    opcode_idx = op.OpcodeIndex()
    opcode = model.OperatorCodes(opcode_idx)
    builtin_code = opcode.BuiltinCode()

    if builtin_code != tflite.BuiltinOperator.CUSTOM:
        op_type = [name for name, val in tflite.BuiltinOperator.__dict__.items() if val == builtin_code][0]
    else:
        op_type = opcode.CustomCode().decode("utf-8")
```

### TFLite Options Extraction (`opt_*.py`)

TFLite stores operator attributes in specific FlatBuffer tables (e.g. `Conv2DOptions`, `ReshapeOptions`, `AddOptions`).

`_get_tflite_attr()` in `src/neuron_toolkit/backends/tflite/utils.py` inspects `BuiltinOptionsType()` and dynamically dispatches to category extractors:

- `opt_activation.py` (`parse_SoftmaxOptions`, `parse_LeakyReluOptions`, etc.)
- `opt_arithmetic.py` (`parse_AddOptions`, `parse_SubOptions`, etc.)
- `opt_nn.py` (`parse_Conv2DOptions`, `parse_FullyConnectedOptions`, etc.)
- `opt_tensor.py` (`parse_ReshapeOptions`, `parse_TransposeOptions`, etc.)
- `opt_quantization` & `opt_sparsity` metadata extraction.

### Quantization & Sparsity Information

Unlike ONNX, TFLite tensors natively contain quantization scales and zero-points:

```python
quantization_info[tensor_name] = {
    "min": min_vals,
    "max": max_vals,
    "scale": scales,
    "zero_point": zero_points,
    "quantized_dimension": quantized_dim,
}
```

This metadata is preserved across queries, pattern matching, and model slicing!
