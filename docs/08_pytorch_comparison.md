# 08 — PyTorch Comparison

Neuron-toolkit can **run a slice of your ONNX/TFLite model** and compare its outputs against a PyTorch reference implementation — giving you numerical similarity metrics.

---

## `graph.compare_outputs()` — Programmatic Comparison

```python
graph.compare_outputs(
    start_points,      # str | list[str] — input tensor names (or node objects)
    end_points,        # str | list[str] — output tensor names (or node objects)
    pytorch_module,    # callable or nn.Module — your PyTorch reference
    inputs_dict,       # dict[str, np.ndarray] — input data
)
# Returns: dict[tensor_name, report_dict]
```

Each `report_dict` contains:

| Key | Type | Description |
|---|---|---|
| `all_close` | `bool` | Whether outputs are numerically close |
| `cosine_similarity` | `float` | Cosine similarity score (0–1) |
| `mae` | `float` | Mean Absolute Error |
| `shape_match` | `bool` | Whether output shapes match |

---

### Single-Output Example (ONNX)

```python
import numpy as np
import torch
from torch import nn
from neuron_toolkit import NeuronGraph

graph = NeuronGraph("complex.onnx")

# Extract weights from graph
weights = graph.get_weights()
B_np = weights["B"]
C_np = weights["C"]

# Define matching PyTorch model
class PyTorchReluAddMul(nn.Module):
    def __init__(self, B, C):
        super().__init__()
        self.B = torch.tensor(B)
        self.C = torch.tensor(C)
    def forward(self, x):
        return (torch.relu(x) + self.B) * self.C

model_py = PyTorchReluAddMul(B_np, C_np)

# Compare slice: from "A_id" to "mul_out"
sample_input = {"A_id": np.random.randn(1, 10).astype(np.float32)}

report = graph.compare_outputs(
    start_points="A_id",
    end_points="mul_out",
    pytorch_module=model_py,
    inputs_dict=sample_input,
)

key = "mul_out"
assert report[key]["all_close"] is True
assert report[key]["cosine_similarity"] > 0.999
assert report[key]["mae"] < 1e-5
```

> Source: `test_slicing.py::test_onnx_pytorch_comparison`

---

### Multi-Output Example

```python
# Model outputs S = A + B and P = A * B
def ref(x, y):
    return x + y, x * y

a = np.random.randn(1, 4).astype(np.float32)
b = np.random.randn(1, 4).astype(np.float32)

report = graph.compare_outputs(["A"], ["S", "P"], ref, {"A": a, "B": b})

assert report["S"]["all_close"] is True
assert report["P"]["all_close"] is True
```

> Source: `test_graph_pytorch.py::test_compare_outputs_multi_output_tuple`

---

### Using Node Objects as Endpoints

```python
end_node = graph.nodes[-1]   # the last node in the graph

report = graph.compare_outputs(["A"], [end_node], ref, {"A": a, "B": b})
assert report["Z"]["all_close"] is True
```

> Source: `test_graph_pytorch.py::test_compare_outputs_object_endpoints`

---

## `@graph.verify_against_pytorch` — Decorator API

Wrap any Python function with this decorator to automatically run the comparison every time the function is called:

```python
@graph.verify_against_pytorch(start_points=["A"], end_points=["Z"])
def ref(x, y):
    return x * y + x + y

# Call like a normal function
a = torch.randn(1, 4)
b = torch.randn(1, 4)
py_out = ref(a, b)           # returns the PyTorch output normally

# After the call, the similarity report is attached to the function
report = ref.similarity_report
print(report["Z"]["all_close"])         # True
print(report["Z"]["shape_match"])       # True
```

### Detecting Mismatches

```python
@graph.verify_against_pytorch(["A"], ["Z"])
def wrong_ref(x, y):
    return x * y - x - y    # intentionally wrong

wrong_ref(a, b)
assert wrong_ref.similarity_report["Z"]["all_close"] is False
assert wrong_ref.similarity_report["Z"]["shape_match"] is True  # shape still matches
```

> Source: `test_graph_pytorch.py::test_verify_against_pytorch_matches`, `test_verify_against_pytorch_detects_mismatch`

---

### Passing Inputs as kwargs

The decorator also handles kwargs:

```python
@graph.verify_against_pytorch(["A"], ["Z"])
def ref(x, y):
    return x * y + x + y

ref(a, y=torch.as_tensor(b))   # works with keyword argument
assert ref.similarity_report["Z"]["all_close"] is True
```

> Source: `test_graph_pytorch.py::test_verify_against_pytorch_kwargs_input`

---

## TFLite Comparison + Decorator

The same API works on TFLite graphs (when the TFLite interpreter is available):

```python
from neuron_toolkit.graph import _is_tflite_runnable
import pytest

if not _is_tflite_runnable():
    pytest.skip("LiteRT interpreter not available")

graph = NeuronGraph("complex.tflite")

class PyTorchAbs(nn.Module):
    def forward(self, x):
        return torch.abs(x)

model_py = PyTorchAbs()
sample_input = {"input": np.random.randn(1, 10).astype(np.float32)}

report = graph.compare_outputs(
    start_points="input",
    end_points="abs_out",
    pytorch_module=model_py,
    inputs_dict=sample_input,
)

key = "abs_out"
assert report[key]["all_close"] is True
assert report[key]["cosine_similarity"] > 0.999
assert report[key]["mae"] < 1e-6

# Using the decorator
@graph.verify_against_pytorch(start_points="input", end_points="abs_out")
def run_layer(x):
    return torch.abs(x)

tensor_input = torch.from_numpy(sample_input["input"])
run_layer(tensor_input)

assert hasattr(run_layer, "similarity_report")
assert run_layer.similarity_report[key]["all_close"] is True
```

> Source: `test_slicing.py::test_pytorch_comparison`

---

## Summary of API Shapes

```python
# Single string endpoint
graph.compare_outputs("A_id", "mul_out", model_py, {"A_id": x})

# List of strings
graph.compare_outputs(["A"], ["S", "P"], ref, {"A": a, "B": b})

# Mix of strings and node objects
graph.compare_outputs(["A"], [graph.nodes[-1]], ref, {"A": a, "B": b})

# Decorator — positional args = input tensor names
@graph.verify_against_pytorch(["A"], ["Z"])
def ref(x, y): ...
```

---

## What's next?

→ [09 — TFLite Backend](./09_tflite_backend.md) — TFLite-specific features and attribute parsing.
