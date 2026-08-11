# Module 09: Optimization Passes (Conv + BN Fusion)

This module explains how graph optimization passes work in `neuron-toolkit`, taking a deep dive into the mathematical folding of `Convolution` and `BatchNormalization` layers (`fuse_conv_bn`).

---

## 1. What is Layer Fusion?

In deep learning inference, `BatchNormalization` is frequently used during training after a `Convolution` layer. During inference, `BatchNormalization` is a linear transformation:

$$\text{BN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Since `Convolution` is also a linear operation ($W \cdot x + b$), both operators can be mathematically merged into a **single fused Convolution operator** ($\hat{W} \cdot x + \hat{b}$).

### Benefits:
- **Reduces Memory Access**: Eliminates intermediate activation tensor allocations between Conv and BN.
- **Speeds Up Inference**: Reduces kernel launch overhead on GPUs/NPUs.

---

## 2. Mathematical Formulation of Conv + BN Fusion

Let the `Convolution` operation be:
$$y_{\text{conv}} = W \cdot x + b_{\text{conv}}$$

Let `BatchNormalization` be:
$$y_{\text{bn}} = \gamma \cdot \frac{y_{\text{conv}} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Substituting $y_{\text{conv}}$ into $y_{\text{bn}}$:
$$y_{\text{bn}} = \gamma \cdot \frac{(W \cdot x + b_{\text{conv}}) - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Expanding terms:
$$y_{\text{bn}} = \left( W \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \right) \cdot x + \left( \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot (b_{\text{conv}} - \mu) + \beta \right)$$

Therefore, the **fused weight** $\hat{W}$ and **fused bias** $\hat{b}$ are:

$$\alpha = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}$$
$$\hat{W} = W \cdot \alpha$$
$$\hat{b} = b_{\text{conv}} \cdot \alpha + \beta - \mu \cdot \alpha$$

*(Note: If Conv has no initial bias $b_{\text{conv}}$, then $\hat{b} = \beta - \mu \cdot \alpha$.)*

---

## 3. Implementation in `GraphPasses` (`src/neuron_toolkit/graph.py`)

`GraphPasses.fuse_conv_bn()` implements this fusion using `PatternDetector` and `NeuronRewriter`:

```python
def fuse_conv_bn(self) -> NeuronRewriter:
    conv = Pattern.op("Conv").capture("conv")
    bn = Pattern.op("BatchNormalization", conv).capture("bn")
    matches = self.graph.findall(bn)
    rewriter = self.graph.rewriter()

    for m in matches:
        conv_node = m.bindings["conv"]
        bn_node = m.bindings["bn"]
        
        # Extract constants: scale (gamma), beta, mean (mu), var (sigma^2), weight (W)
        scale = _const(bn_inputs[1])
        beta  = _const(bn_inputs[2])
        mean  = _const(bn_inputs[3])
        var   = _const(bn_inputs[4])
        weight= _const(conv_inputs[1])
        eps   = float(bn_attrs.get("epsilon", 1e-5))

        # Compute fused scale multiplier
        multiplier = (scale / np.sqrt(var + eps)).astype(weight.dtype)
        scale_b = multiplier.reshape((scale.shape[0],) + (1,) * (weight.ndim - 1))
        
        # Compute fused weights and bias
        new_weight = (weight * scale_b).astype(weight.dtype)
        if len(conv_inputs) > 2 and conv_inputs[2]:
            conv_bias = _const(conv_inputs[2])
            new_bias = (conv_bias * multiplier + beta - mean * multiplier).astype(weight.dtype)
        else:
            new_bias = (beta - mean * multiplier).astype(weight.dtype)

        # Register new initializers in rewriter
        fused_w = f"{bn_name}_weight"
        fused_b = f"{bn_name}_bias"
        rewriter.register_initializer(fused_w, new_weight)
        rewriter.register_initializer(fused_b, new_bias)

        # Replace Conv + BN with single Fused Conv node
        rewriter.replace(
            [conv_node, bn_node],
            "Conv",
            inputs=[conv_inputs[0], fused_w, fused_b],
            outputs=list(bn_node.output),
            **backend.get_node_attrs(conv_node)
        )
    return rewriter
```

---

## 4. Safety Guards & Edge Cases

The fusion pass checks critical safety conditions before modifying the graph:
1. **Training Mode Check**: If `training_mode != 0` (model is still training), fusion is skipped because statistics are not static.
2. **Dimension Matching**: Verifies `weight.shape[0] == scale.shape[0]`.
3. **Missing Constants**: If any parameter tensor (`scale`, `beta`, `mean`, `var`, `weight`) is dynamic (not in `tensor_map`), fusion is skipped.
