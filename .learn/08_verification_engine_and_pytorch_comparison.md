# Module 08: Verification Engine & PyTorch Cross-Comparison

This module explains how `neuron-toolkit` performs numerical verification (`compare_outputs` and `@verify_against_pytorch`) by executing sliced subgraphs and comparing output tensors against native PyTorch modules.

---

## 1. Why Cross-Verification is Essential

When performing graph optimizations (such as operator fusion, quantization, or node replacements), subtle numerical bugs can arise due to:
- Mismatched attribute configurations (padding, strides, dilations).
- Precision loss during parameter conversion.
- Operator semantics differences between PyTorch, ONNX, and TFLite.

`neuron-toolkit` provides a **built-in cross-verification engine** that runs PyTorch and backend runtimes (ONNX Runtime or LiteRT) on identical inputs and calculates precision metrics.

---

## 2. Theoretical Metrics

To evaluate numerical fidelity between PyTorch outputs $Y_{\text{torch}}$ and sliced backend outputs $Y_{\text{model}}$:

### 1. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |Y_{\text{torch}}[i] - Y_{\text{model}}[i]|$$

### 2. Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (Y_{\text{torch}}[i] - Y_{\text{model}}[i])^2$$

### 3. Cosine Similarity
$$\text{CosineSim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
Measures structural direction alignment between flattened output vectors (value near $1.0$ indicates identical shape dynamics).

### 4. `allclose` Guard
Uses `numpy.allclose(Y_torch, Y_model, rtol=1e-5, atol=1e-8)` to check if every element satisfies absolute and relative tolerances.

---

## 3. Sliced Execution & Comparison Pipeline (`compare_outputs`)

Inside `src/neuron_toolkit/graph.py`:

```
Step 1: Execute PyTorch Module
  ├── Convert numpy input dict to torch.Tensors
  ├── Run torch_module(*torch_inputs) with torch.no_grad()
  └── Convert PyTorch outputs back to numpy arrays

Step 2: Execute Sliced Model Subgraph
  ├── Automatically slice model between start_points and end_points
  ├── If ONNX Backend: Serialize sliced model -> Run onnxruntime.InferenceSession
  └── If TFLite Backend: Run LiteRT / ai_edge_litert.interpreter.Interpreter

Step 3: Align Inputs & Outputs
  ├── Name-matching fallback
  └── Positional fallback matching

Step 4: Calculate Metrics (MAE, MSE, Cosine Sim, allclose)
  └── Return comprehensive report dictionary
```

---

## 4. The `@verify_against_pytorch` Decorator

You can decorate PyTorch forward functions or layer calls to automatically test model subgraphs:

```python
import torch
import torch.nn as nn
from neuron_toolkit import NeuronGraph

graph = NeuronGraph.load("conv_model.onnx")
conv_module = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

# Attach verification decorator:
@graph.verify_against_pytorch(
    start_points="conv_input",
    end_points="conv_output",
    rtol=1e-4,
    atol=1e-5
)
def run_layer(x):
    return conv_module(x)

# Executing the function runs BOTH PyTorch and ONNX, populating .similarity_report:
sample_input = torch.randn(1, 3, 224, 224)
output = run_layer(sample_input)

# Inspect verification results:
print(run_layer.similarity_report)
# Output:
# {
#   "conv_output": {
#       "mae": 1.2e-7,
#       "mse": 3.4e-14,
#       "cosine_similarity": 0.9999999,
#       "all_close": True,
#       "shape_match": True
#   }
# }
```

In the next module, we will examine **Optimization Passes and Conv-BN Fusion**.
