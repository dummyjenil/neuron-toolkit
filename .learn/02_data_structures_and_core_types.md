# Module 02: Data Structures & Core Types

In this module, you will learn about the foundational data structures and core type abstractions in `neuron-toolkit`.

---

## 1. Type Aliases (`_types.py`)

To keep function signatures clean and readable, `neuron-toolkit` defines type aliases in `src/neuron_toolkit/_types.py`:

```python
import numpy as np

TensorMap = dict[str, np.ndarray]
```

`TensorMap` is a key-value dictionary where:
- **Key**: Name of a model parameter, weight, or initializer (e.g. `"conv1.weight"`).
- **Value**: A multi-dimensional `numpy.ndarray` containing numeric values.

---

## 2. Shape & Data Type Metadata (`ShapeInfo`)

Throughout the library, shape and data type information for intermediate tensors and graph inputs/outputs is tracked using `ShapeInfo`:

```python
# ShapeInfo maps value_name -> (rank | None, dtype_str | None)
ShapeInfo = dict[str, tuple[int | None, str | None]]
```

### Example:
```python
{
    "input_0": (4, "float32"),       # Rank 4 tensor [N, C, H, W], float32
    "conv1_out": (4, "float32"),
    "labels": (1, "int64"),          # Rank 1 tensor, int64
}
```

---

## 3. The Unified Entry Point (`NeuronGraph`)

Defined in `src/neuron_toolkit/graph.py`, `NeuronGraph` is the public façade wrapping backend parsers (`ONNXParser` or `TFLiteParser`).

### Key Properties & Methods:

```python
class NeuronGraph:
    def __init__(self, source: object, **kwargs: Any) -> None: ...
```

#### Initialization Logic:
`NeuronGraph` supports auto-detection of inputs:
1. **`str` path**: Calls `NeuronGraph.load(path)`.
2. **`bytes`**: Inspects flatbuffer magic header at offset 4 (`source[4:8] == b"TFL3"`). If matched, uses `TFLiteParser`. Otherwise, uses `ONNXParser`.
3. **ONNX `ModelProto`**: Uses `ONNXParser`.
4. **TFLite `Model` object**: Uses `TFLiteParser`.

#### Core Properties:
- `nodes -> list[object]`: Returns all operators in the model.
- `tensor_map -> dict[str, object]`: Returns all initializers / weight tensors.
- `shape_info -> ShapeInfo`: Returns shape rank and data type annotations.

---

## 4. `_GraphShim`: The Internal Stand-in

In `src/neuron_toolkit/_utils.py`, `_GraphShim` serves as a lightweight stand-in for a full model when running internal tools like `PatternDetector`:

```python
class _GraphShim:
    """Lightweight stand-in for ModelProto used inside PatternDetector."""

    def __init__(
        self,
        nodes: Sequence[object],
        tensor_map: Mapping[str, object],
        shape_info: ShapeInfo | None = None,
        backend: object | None = None,
    ) -> None:
        self.nodes = nodes
        self.tensor_map = tensor_map
        self.shape_info: ShapeInfo = shape_info or {}
        self.backend = backend
```

Why use `_GraphShim`?
When matching patterns on subgraphs or sliced node lists, we don't need a full heavy `ModelProto` or TFLite FlatBuffer object. A simple `_GraphShim` holding `nodes`, `tensor_map`, and `shape_info` is sufficient for graph traversal algorithms!

---

## 5. Node Abstraction (`TFLiteNode` vs `NodeProto`)

ONNX models natively use `onnx.NodeProto` objects, which have:
- `op_type`: String (e.g., `"Conv"`, `"Relu"`).
- `name`: String node identifier.
- `input`: List of input tensor name strings.
- `output`: List of output tensor name strings.
- `attribute`: List of `AttributeProto` objects.

TFLite flatbuffers do not have an explicit `NodeProto` class. To make TFLite nodes compatible with ONNX query algorithms, `TFLiteParser` creates a mock data structure called `TFLiteNode` (`src/neuron_toolkit/backends/tflite/parser.py`):

```python
@dataclass
class TFLiteNode:
    """Mock-up of a TFLite node that looks like an ONNX node for the query engine."""

    op_type: str
    name: str
    input: list[str]
    output: list[str]
    attrs: dict[str, object] = field(default_factory=dict)
    op_idx: int | None = None
```

Because both `NodeProto` and `TFLiteNode` expose `.op_type`, `.name`, `.input`, `.output`, and attributes via backend helpers, the entire query and pattern matching engine works identically across both backends!

---

## 6. `MatchResult` (`src/neuron_toolkit/pattern/models.py`)

When `PatternDetector` finds a match in the graph, it returns a `MatchResult`:

```python
class MatchResult(BaseModel):
    start: object                           # The root matched node
    end: object                             # The terminal matched node
    nodes: list[object]                     # All nodes in the matched subgraph
    bindings: dict[str, object]             # Named captured nodes (e.g., {"conv": conv_node})
    tensor_map: TensorMap                   # Reference to weight tensors

    def as_query(self) -> NeuronQuery:
        """Return the matched subgraph as a NeuronQuery."""
        from neuron_toolkit.query import NeuronQuery
        return NeuronQuery(list(self.nodes), self.tensor_map, list(self.nodes))
```

This allows you to convert any match directly into a `NeuronQuery` for further inspection or rewriting.
