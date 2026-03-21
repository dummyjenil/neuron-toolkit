# Getting Started

## Installation

`onnx_toolkit` requires Python 3.8 or later.

Install the required dependencies:

```bash
pip install onnx numpy
```

Install optional dependencies for export features:

```bash
pip install pandas       # required for ONNXQuery.to_dataframe()
```

Place the `onnx_toolkit` package directory on your Python path, or install it as a local package:

```bash
pip install -e /path/to/onnx_toolkit
```

---

## Loading a Model

Every workflow begins with `ONNXParser`:

```python
from onnx_toolkit import ONNXParser

parser = ONNXParser("model.onnx")
print(parser.summary())
```

Example output:

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
    BatchNormalization          12
    ...
```

Shape inference runs automatically on load. It is required for `Pattern.with_output_rank()` and `Pattern.with_dtype()` constraints to work. To disable it:

```python
parser = ONNXParser("model.onnx", infer_shapes=False)
```

---

## Your First Query

`parser.find()` returns an `ONNXQuery` over all graph nodes. Every method on `ONNXQuery` returns a new `ONNXQuery`, so calls chain freely:

```python
# All Conv nodes that have weight tensors
convs = parser.find().find_by_op_type("Conv").has_params()
print(convs.count(), "Conv nodes with weights")
print(convs)
```

Example `repr` output:

```
ONNXQuery: 48 node(s)
  [  0] Conv             /layer1/conv1
         params: weight:[64, 3, 7, 7], bias:[64]
  [  1] Conv             /layer1/conv2
         params: weight:[64, 64, 3, 3], bias:[64]
  ...
```

---

## Your First Pattern Match

Patterns describe the *shape* of a subgraph. Build a pattern, then ask the query which nodes match it:

```python
from onnx_toolkit import Pattern

x = Pattern.any().capture("x")

# Find every SiLU (x * sigmoid(x)) in the graph
results = parser.find().match_results(Pattern.silu(x))

for r in results:
    print(f"SiLU at {r.start.name}, input = {r.bindings['x'].name}")
```

---

## Your First Rewrite

Use `GraphRewriter` to replace matched subgraphs:

```python
rw = parser.rewriter()

for r in parser.find().match_results(Pattern.gelu(x)):
    rw.replace_from_result(r, new_op="Gelu")

new_model = rw.build("model_gelu_fused.onnx")
print(f"Saved rewritten model with {len(new_model.graph.node)} nodes")
```

---

## Enabling Debug Logging

`onnx_toolkit` uses Python's standard `logging` module. Four sub-loggers are available:

| Logger | Covers |
|---|---|
| `onnx_toolkit` | Parser load, summary |
| `onnx_toolkit.query` | Query filter and traversal steps |
| `onnx_toolkit.pattern` | Pattern construction |
| `onnx_toolkit.detect` | PatternDetector DFS walk and decisions |

Enable all of them at once:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Enable only the DFS tracer for pattern debugging:

```python
import logging
logging.getLogger("onnx_toolkit.detect").setLevel(logging.DEBUG)
logging.getLogger("onnx_toolkit.detect").addHandler(logging.StreamHandler())
```

---

## Package Layout

```
onnx_toolkit/
├── __init__.py      Public re-exports: ONNXParser, ONNXQuery, Pattern,
│                    PatternDetector, MatchResult, GraphRewriter, TensorMap
├── parser.py        ONNXParser
├── query.py         ONNXQuery
├── pattern.py       Pattern, PatternDetector, MatchResult
├── rewriter.py      GraphRewriter
├── _utils.py        _attr_value, _node_attrs, _GraphShim, _build_shape_info
└── _types.py        TensorMap type alias
```

All public names are importable directly from `onnx_toolkit`:

```python
from onnx_toolkit import ONNXParser, ONNXQuery, Pattern, PatternDetector
from onnx_toolkit import MatchResult, GraphRewriter, TensorMap
```
