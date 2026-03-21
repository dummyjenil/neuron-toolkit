# ONNXParser

`ONNXParser` loads an ONNX model from disk and exposes its graph for querying, pattern detection, and rewriting. It is the single entry point for all other library components.

---

## Constructor

```python
ONNXParser(onnx_path: str, *, infer_shapes: bool = True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `onnx_path` | `str` | — | Path to the `.onnx` file |
| `infer_shapes` | `bool` | `True` | Run `onnx.shape_inference.infer_shapes` on load |

Shape inference annotates every intermediate tensor with its rank and dtype. It is required for `Pattern.with_output_rank()` and `Pattern.with_dtype()` constraints to function. If inference fails (e.g. due to dynamic shapes), a warning is logged and the model is still loaded without annotations.

```python
from onnx_toolkit import ONNXParser

parser = ONNXParser("resnet50.onnx")
parser = ONNXParser("dynamic_model.onnx", infer_shapes=False)
```

---

## Attributes

After construction the following attributes are available directly:

| Attribute | Type | Description |
|---|---|---|
| `model` | `onnx.ModelProto` | The raw ONNX model (inferred if `infer_shapes=True`) |
| `nodes` | `list[NodeProto]` | All nodes in graph order |
| `tensor_map` | `dict[str, np.ndarray]` | Initializer tensors keyed by name |
| `graph_inputs` | `set[str]` | Names of graph-level input tensors |
| `graph_outputs` | `set[str]` | Names of graph-level output tensors |
| `shape_info` | `dict[str, (int\|None, str\|None)]` | Per-output `(rank, dtype)` from shape inference |

```python
print(parser.nodes[0].op_type)         # first node's op type
print(parser.tensor_map["conv1.weight"].shape)
print(parser.graph_inputs)             # {"input"}
print(parser.shape_info.get("relu_0")) # (4, "float32")
```

---

## Methods

### `find() → ONNXQuery`

Returns an `ONNXQuery` over **all** nodes in the graph. This is the starting point for every query chain.

```python
all_nodes = parser.find()
convs     = parser.find().find_by_op_type("Conv")
```

---

### `pattern_detect(pattern, start_node, end_node=None) → MatchResult | None`

Creates a `PatternDetector` bound to this model and immediately calls `.match()`. A convenience shortcut for one-off checks against a named node.

| Parameter | Type | Description |
|---|---|---|
| `pattern` | `Pattern` | The pattern to match |
| `start_node` | `str \| NodeProto` | Node name or object where the match begins |
| `end_node` | `str \| NodeProto \| None` | Optional exclusive upstream boundary |

Returns a `MatchResult` on success, `None` on failure. See [PatternDetector & MatchResult](detector.md) for the full end-node contract.

```python
x   = Pattern.any().capture("x")
r   = parser.pattern_detect(Pattern.gelu(x), start_node="Mul_42")

if r:
    print("GELU found, input node:", r.bindings["x"].name)
```

---

### `rewriter() → GraphRewriter`

Returns a `GraphRewriter` bound to this parser. The original model is never modified; all edits are staged until `.build()` is called.

```python
rw = parser.rewriter()
rw.delete(parser.find().find_by_op_type("Dropout").nodes)
rw.build("no_dropout.onnx")
```

---

### `summary() → str`

Returns a human-readable text summary of the model: node count, tensor count, graph inputs/outputs, shape annotation count, and a ranked frequency table of op types.

```python
print(parser.summary())
```

```
ONNX model summary
  Nodes        : 147
  Tensors      : 102
  Graph inputs : 1
  Graph outputs: 1
  Shape info   : 289 annotated tensors
  Op types (12):
    Conv                        48
    Relu                        24
    Add                         18
```

---

## Shape Info Format

`parser.shape_info` is a `dict[str, tuple[int | None, str | None]]` mapping each annotated tensor name to `(rank, dtype)`:

- `rank` is an integer (e.g. `4` for `NCHW`) or `None` if unknown.
- `dtype` is a numpy dtype string such as `"float32"`, `"float16"`, `"int64"`, or `None` if unknown.

```python
rank, dtype = parser.shape_info.get("features", (None, None))
print(rank, dtype)  # 4  float32
```

Supported dtype mappings:

| ONNX type | numpy string |
|---|---|
| `FLOAT` | `"float32"` |
| `DOUBLE` | `"float64"` |
| `FLOAT16` | `"float16"` |
| `INT8` / `INT16` / `INT32` / `INT64` | `"int8"` … `"int64"` |
| `UINT8` … `UINT64` | `"uint8"` … `"uint64"` |
| `BOOL` | `"bool"` |
| `STRING` | `"object"` |
