# Module 10: Rebuilding `neuron-toolkit` from Scratch

This module is your step-by-step practical guide to rebuilding `neuron-toolkit` completely from scratch in a fresh workspace!

---

## Step 1: Project Setup & Package Configuration

Create a project directory structure and configure build metadata using `pyproject.toml` (Hatchling backend):

```bash
mkdir -p neuron-toolkit/src/neuron_toolkit/{backends/{onnx,tflite},pattern,query}
mkdir -p neuron-toolkit/tests
cd neuron-toolkit
```

### `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "neuron-toolkit"
version = "0.1.0"
dependencies = [
    "onnx",
    "numpy",
    "pydantic",
    "networkx",
]
```

---

## Step 2: Implement Base Types & Utilities

1. **`src/neuron_toolkit/_types.py`**: Define `TensorMap = dict[str, np.ndarray]`.
2. **`src/neuron_toolkit/_utils.py`**:
   - Implement `_GraphShim(nodes, tensor_map, shape_info, backend)`.
   - Implement `trace_subgraph_boundaries(nodes, tensor_map, start_points, end_points)`.

---

## Step 3: Implement Abstract Base Backend Classes

Create `src/neuron_toolkit/backends/base.py`:

- `BaseParser`: Defines abstract methods (`find()`, `pattern_detect()`, `rewriter()`, `get_node_attrs()`, `is_constant_node()`, `get_constant_value()`, `summary()`).
- `BaseRewriter`: Defines abstract methods (`replace()`, `delete()`, `insert_before()`, `reset()`, `build()`).

---

## Step 4: Implement ONNX Backend

1. **`backends/onnx/utils.py`**: Attribute extractor `_node_attrs()` and shape builder `_build_shape_info()`.
2. **`backends/onnx/parser.py`**: `LazyTensorMap` initializer map and `ONNXParser`.
3. **`backends/onnx/rewriter.py`**: `ONNXRewriter` with NetworkX topological sort.

---

## Step 5: Implement TFLite Backend

1. **`backends/tflite/utils.py`**: `_TFLITE_DTYPE_TO_NP` mapping and option parsers.
2. **`backends/tflite/opt_*.py`**: Category attribute parsers (`opt_activation.py`, `opt_arithmetic.py`, `opt_nn.py`, `opt_tensor.py`).
3. **`backends/tflite/parser.py`**: `TFLiteNode` data structure and `TFLiteParser` FlatBuffer reader.
4. **`backends/tflite/rewriter.py`**: `TFLiteRewriter` with FlatBuffer builder table cloners (`_copy_flatbuffer_table`, `_copy_quantization`).

---

## Step 6: Implement Fluent Query Engine

Create `src/neuron_toolkit/query/core.py`:

- Implement `NeuronQuery`.
- Add `@cached_property` graph structures (`output_map`, `_nx_graph`, `_node_to_idx`).
- Implement filters (`op`, `name`, `attr`, `rank`, `dtype`, `find_by_param_name`).
- Implement graph traversals (`parents`, `children`, `ancestors`, `descendants`).
- Implement set operations (`union`, `intersection`, `difference`).

---

## Step 7: Implement Pattern Matching DSL & Detector

1. **`pattern/models.py`**: `MatchResult(start, end, nodes, bindings, tensor_map)` with `.as_query()`.
2. **`pattern/_activations.py`**: `ActivationMixin` factory methods (`relu`, `silu`, `gelu`, `mish`, `relu6`).
3. **`pattern/dsl.py`**: `Pattern` class with `.op()`, `.any()`, `.const()`, `.any_of()`, `.capture()`, `.where()`, and arithmetic operator overloads.
4. **`pattern/_matcher.py`**: `MatchContext` and `MatchingMixin` DFS algorithm with memoization, backtracking, referential consistency, and commutative permutations.
5. **`pattern/detector.py`**: `PatternDetector` wrapper.

---

## Step 8: Implement Staged Rewriter Façade & Unified Entry Point

1. **`src/neuron_toolkit/rewriter.py`**: `NeuronRewriter` delegating calls to `BaseRewriter`.
2. **`src/neuron_toolkit/graph.py`**: `NeuronGraph` entry point, `load()`, `match()`, `findall()`, `slice()`, `compare_outputs()`, `@verify_against_pytorch`, and `GraphPasses.fuse_conv_bn()`.
3. **`src/neuron_toolkit/__init__.py`**: Re-export public API.

---

## Step 9: Verification & Testing

Run pytest across your new implementation:

```bash
pytest --cov=src --cov-report=term-missing
```

Congratulations! You have successfully mastered and rebuilt the entire `neuron-toolkit` codebase from first principles!
