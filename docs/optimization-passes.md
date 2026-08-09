# Graph Optimization Passes (`GraphPasses`)

`GraphPasses` provides graph optimization and fusion transformations in **Neuron-Toolkit**. It is accessed through the `.passes` property on `NeuronGraph`.

---

## Accessing Optimization Passes

```python
from neuron_toolkit import NeuronGraph

graph = NeuronGraph.load("model.onnx")
passes = graph.passes
```

---

## Built-In Passes

### 1. Conv + BatchNormalization Folding (`fuse_conv_bn`)

```python
rewriter = graph.passes.fuse_conv_bn()
fused_graph = rewriter.build(output_path="fused.onnx")
```

#### Transformation Details

`fuse_conv_bn()` searches for patterns where a `Conv` operator is directly followed by a `BatchNormalization` operator:

$$\text{Pattern: } \text{Conv}(X, W, B) \rightarrow \text{BatchNormalization}(Y, \gamma, \beta, \mu, \sigma^2)$$

The pass folds the BatchNormalization scaling factor $\gamma$, shift $\beta$, running mean $\mu$, running variance $\sigma^2$, and epsilon $\epsilon$ directly into the Conv weight tensor $W$ and bias tensor $B$:

$$\gamma_{\text{scale}} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}$$

$$W_{\text{fused}} = W \cdot \gamma_{\text{scale}}$$

$$B_{\text{fused}} = (B - \mu) \cdot \gamma_{\text{scale}} + \beta$$

#### Benefits

- Replaces 2 operator nodes (`Conv` and `BatchNormalization`) with 1 fused `Conv` node.
- Eliminates memory read/write overhead of intermediate normalization tensor.
- Registers $W_{\text{fused}}$ and $B_{\text{fused}}$ as new initializer tensors via `rewriter.register_initializer()`.

---

## Custom Optimization Passes

You can build custom graph optimization passes using `Pattern` matching and `NeuronRewriter`:

```python
from neuron_toolkit import NeuronGraph, Pattern

def fuse_add_relu_pass(graph: NeuronGraph):
    # 1. Define target subgraph pattern
    add = Pattern.op("Add").capture("add")
    relu = Pattern.relu(add).capture("relu")

    # 2. Find all matching subgraphs
    matches = graph.findall(relu)
    rewriter = graph.rewriter()

    # 3. Stage transformations
    for match in matches:
        add_node = match.bindings["add"]
        inputs = list(add_node.input)
        outputs = list(match.bindings["relu"].output)
        
        # Replace Add + Relu with single fused op
        rewriter.replace(
            nodes=match.nodes,
            new_op="FusedAddRelu",
            inputs=inputs,
            outputs=outputs
        )

    # 4. Apply edits
    return rewriter.build()
```
