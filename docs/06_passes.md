# 06 — Built-in Passes

`graph.passes` exposes ready-made graph optimization passes you can run with a single call.

---

## Accessing Passes

```python
from neuron_toolkit import NeuronGraph

graph = NeuronGraph("model.onnx")
graph.passes          # the passes namespace object
```

Each pass returns a **`NeuronRewriter`** with edits already staged. Call `.build()` to materialize.

---

## `graph.passes.fuse_conv_bn()` — Fold BatchNormalization into Conv

This pass folds a `BatchNormalization` node immediately following a `Conv` node into the Conv's weights and bias — resulting in a single `Conv` with no BN overhead.

### Basic Usage

```python
import numpy as np

graph = NeuronGraph("conv_bn_model.onnx")
assert [n.op_type for n in graph.nodes] == ["Conv", "BatchNormalization"]

fused = graph.passes.fuse_conv_bn().build()

# The BN is now gone
assert [n.op_type for n in fused.graph.node] == ["Conv"]

# New merged weights are present
initializer_names = {i.name for i in fused.graph.initializer}
assert "bn0_weight" in initializer_names
assert "bn0_bias" in initializer_names
```

### Verify Numerical Correctness

```python
import onnxruntime as ort
import numpy as np

def run(model_path, x):
    session = ort.InferenceSession(str(model_path))
    return session.run(None, {"X": x})[0]

x = np.random.randn(1, 1, 5, 5).astype(np.float32)

original_out = run("conv_bn.onnx", x)
fused_out    = run("fused.onnx",   x)

np.testing.assert_allclose(original_out, fused_out, atol=1e-6)
```

> Source: `test_onnx_passes.py::test_fuse_conv_bn_folds_to_single_conv`

---

### Works With or Without Conv Bias

```python
# Conv with bias
graph = NeuronGraph("conv_bn_with_bias.onnx")
fused = graph.passes.fuse_conv_bn().build()
assert [n.op_type for n in fused.graph.node] == ["Conv"]

# Conv without bias — also works
graph = NeuronGraph("conv_bn_no_bias.onnx")
fused = graph.passes.fuse_conv_bn().build()
assert [n.op_type for n in fused.graph.node] == ["Conv"]
```

> Source: `test_onnx_passes.py::test_fuse_conv_bn_without_conv_bias`

---

### Works With Grouped Convolutions

```python
graph = NeuronGraph("conv_bn_grouped.onnx")   # Conv with group=2
fused = graph.passes.fuse_conv_bn().build()

conv_node = fused.graph.node[0]
# The group attribute is preserved
assert any(a.name == "group" and a.i == 2 for a in conv_node.attribute)
```

> Source: `test_onnx_passes.py::test_fuse_conv_bn_grouped_conv`

---

### Skips Training-Mode BatchNormalization

If the BN node has `training_mode=1`, the pass will **not** fuse it (it would be incorrect):

```python
graph = NeuronGraph("conv_bn_training.onnx")   # BN has training_mode=1
rewriter = graph.passes.fuse_conv_bn()

# build() raises because no fusion was staged
import pytest
with pytest.raises(ValueError, match="No edits staged"):
    rewriter.build()

# Graph is unchanged
assert [n.op_type for n in graph.nodes] == ["Conv", "BatchNormalization"]
```

> Source: `test_onnx_passes.py::test_fuse_conv_bn_training_mode_skips`

---

### Raises If No Conv→BN Pattern Found

```python
graph = NeuronGraph("identity_only.onnx")   # no Conv node
rewriter = graph.passes.fuse_conv_bn()

with pytest.raises(ValueError, match="No edits staged"):
    rewriter.build()
```

> Source: `test_onnx_passes.py::test_fuse_conv_bn_no_match_raises`

---

## `graph.passes` is Callable & Extensible

```python
# The passes namespace always exists
assert hasattr(graph, "passes")
assert callable(graph.passes.fuse_conv_bn)

# It returns a NeuronRewriter
from neuron_toolkit.rewriter import NeuronRewriter
result = graph.passes.fuse_conv_bn()
assert isinstance(result, NeuronRewriter)
```

> Source: `test_fusion.py::test_graph_passes`

---

## Full End-to-End Example

```python
import numpy as np
import onnx
import onnxruntime as ort
from neuron_toolkit import NeuronGraph

# 1. Load model with Conv → BN
graph = NeuronGraph("conv_bn.onnx")
print([n.op_type for n in graph.nodes])   # ['Conv', 'BatchNormalization']

# 2. Run the fusion pass
fused_model = graph.passes.fuse_conv_bn().build()

# 3. Validate the fused ONNX is correct
onnx.checker.check_model(fused_model)
print([n.op_type for n in fused_model.graph.node])  # ['Conv']

# 4. Save
fused_bytes = fused_model.SerializeToString()
open("fused.onnx", "wb").write(fused_bytes)

# 5. Verify numerical equivalence
x = np.random.randn(1, 1, 5, 5).astype(np.float32)

def run(path, x):
    return ort.InferenceSession(path).run(None, {"X": x})[0]

np.testing.assert_allclose(run("conv_bn.onnx", x), run("fused.onnx", x), atol=1e-6)
print("✅ Fused model is numerically identical!")
```

---

## What's next?

→ [07 — Graph Slicing](./07_slicing.md) — extract subgraphs between two tensor boundaries.
