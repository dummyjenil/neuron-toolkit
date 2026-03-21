# Cookbook

This document contains complete, worked examples for the most common `onnx_toolkit` use cases. Each recipe is self-contained and can be adapted directly.

---

## 1. Model Inspection

### Print a model summary

```python
from onnx_toolkit import ONNXParser

parser = ONNXParser("model.onnx")
print(parser.summary())
```

### List all unique op types and their counts

```python
from collections import Counter

counts = Counter(n.op_type for n in parser.nodes)
for op, count in counts.most_common():
    print(f"  {op:<30} {count}")
```

### Print all Conv weight shapes

```python
parser.find().find_by_op_type("Conv").apply(
    lambda n, p: print(n.name, {k: list(v.shape) for k, v in p.items()})
)
```

### Inspect a specific node by name

```python
node = parser.find().find_by_name("layer1/conv1", exact=True).single_node
print(node.op_type)
for attr in node.attribute:
    print(f"  {attr.name}: ...")
```

### Export node metadata to a spreadsheet

```python
df = parser.find().to_dataframe()
df.to_csv("model_nodes.csv", index=False)
```

---

## 2. Graph Traversal

### Find all nodes that directly consume a named tensor

```python
consumers = parser.find().find_by_tensor("features")
print(consumers.op_types())
```

### Find all nodes that feed into a specific node

```python
target    = parser.find().find_by_name("classifier/linear", exact=True)
ancestors = target.ancestors()
print(f"{ancestors.count()} upstream nodes: {ancestors.op_types()}")
```

### Find all nodes downstream of the first Conv

```python
first_conv  = parser.find().find_by_op_type("Conv")[0]
descendants = first_conv.descendants(max_depth=5)
print(descendants)
```

### Find all entry nodes (nodes that read graph inputs)

```python
entries = parser.find().entry_nodes
print("Entry nodes:", [n.op_type for n in entries])
```

### Find all output-producing nodes

```python
outputs = parser.find().output_nodes
print("Output nodes:", [n.name for n in outputs])
```

---

## 3. Filtering

### All depthwise convolutions

```python
dw_convs = parser.find().find_by_attribute("group", lambda g: g > 1)
print(f"{dw_convs.count()} depthwise convolutions")
```

### All dilated convolutions

```python
dilated = parser.find().find_by_attribute(
    "dilations", lambda d: any(v > 1 for v in d)
)
```

### All Conv nodes without a bias

```python
all_conv  = parser.find().find_by_op_type("Conv")
with_bias = parser.find().find_by_op_type("Conv").find_by_param_name("bias")
no_bias   = all_conv.difference(with_bias)
print(f"{no_bias.count()} Conv nodes without bias")
```

### All trainable nodes (Conv, Gemm, MatMul with weights)

```python
trainable = (
    parser.find()
    .find_by_op_type("Conv")
    .union(parser.find().find_by_op_type("Gemm"))
    .union(parser.find().find_by_op_type("MatMul"))
    .has_params()
)
```

---

## 4. Weight Inspection

### Retrieve a weight tensor by name

```python
w = parser.find().tensor("layer1.conv.weight")
print(w.shape, w.dtype)
```

### Count total parameters

```python
import numpy as np

total = sum(v.size for v in parser.tensor_map.values())
print(f"Total parameters: {total:,}")
```

### Find all Conv layers with large kernels

```python
def large_kernel(node, params):
    w = params.get("weight")
    if w is not None and max(w.shape[2:]) >= 7:
        print(node.name, w.shape)

parser.find().find_by_op_type("Conv").apply(large_kernel)
```

### Check for zero-initialised biases

```python
import numpy as np

def check_zero_bias(node, params):
    b = params.get("bias")
    if b is not None and np.all(b == 0):
        print(f"Zero bias: {node.name}")

parser.find().has_params().apply(check_zero_bias)
```

---

## 5. Basic Pattern Matching

### Detect whether a model contains GELU

```python
from onnx_toolkit import Pattern

x    = Pattern.any()
hits = parser.find().matches(Pattern.gelu(x))
print(f"GELU found: {hits.count()} occurrence(s)")
```

### Detect SiLU / Swish activations

```python
x    = Pattern.any().capture("x")
hits = parser.find().match_results(Pattern.silu(x))

for r in hits:
    print(r.start.name, "— input:", r.bindings["x"].name)
```

### Detect any activation function

```python
x   = Pattern.any()
act = Pattern.any_of(
    Pattern.relu(x),
    Pattern.gelu(x),
    Pattern.silu(x),
    Pattern.tanh(x),
    Pattern.sigmoid(x),
)
parser.find().matches(act).apply(lambda n, _: print(n.op_type, n.name))
```

### Detect depthwise conv followed by BatchNorm

```python
dw   = Pattern.op("Conv").where(group=lambda g: g > 1).capture("conv")
bn   = Pattern.op("BatchNormalization").capture("bn")

# PatternDetector with start at BN and end at DW lets us match the pair
x       = Pattern.any()
results = parser.find().find_by_op_type("BatchNormalization").match_results(
    Pattern.op("BatchNormalization", dw)
)

for r in results:
    print("DW conv:", r.bindings["conv"].name)
    print("BN:     ", r.bindings["bn"].name if "bn" in r.bindings else r.start.name)
```

---

## 6. Advanced Pattern Matching

### Match with attribute and shape constraints

```python
# 4-D fp32 depthwise conv only
dw_fp32 = (
    Pattern.op("Conv")
    .where(group=lambda g: g > 1)
    .with_output_rank(4)
    .with_dtype("float32")
    .capture("dw")
)

results = parser.find().match_results(dw_fp32)
print(f"Found {len(results)} matching depthwise convolutions")
```

### Use end_node to scope a match

```python
from onnx_toolkit import PatternDetector

# Match the SiLU at Mul_7 but stop at Add_3
det    = PatternDetector(parser.model, start_node="Mul_7", end_node="Add_3")
result = det.match(Pattern.silu(Pattern.any()))

if result:
    print("Matched from", result.start.name, "to boundary", result.end.name)
    print("Nodes visited:", [n.name for n in result.nodes])
```

### Inspect the matched subgraph

```python
x       = Pattern.any().capture("x")
results = parser.find().match_results(Pattern.gelu(x))

for r in results:
    sub = r.as_query()
    print(f"\nGELU subgraph at {r.start.name}:")
    print(f"  Ops: {sub.op_types()}")
    print(f"  Nodes: {sub.count()}")
    print(f"  Input: {r.bindings['x'].name}")
    sub.to_onnx(f"gelu_{r.start.name}.onnx")
```

---

## 7. Graph Visualisation

### Visualise the full graph

```python
dot = parser.find().to_dot()

with open("full_graph.dot", "w") as f:
    f.write(dot)
# dot -Tsvg full_graph.dot -o full_graph.svg
```

### Visualise a subgraph with weights shown

```python
# Show only the first 3 layers with their weight tensors
sub_dot = parser.find()[:10].to_dot(show_params=True)
with open("sub.dot", "w") as f:
    f.write(sub_dot)
```

### Visualise a matched subgraph

```python
x       = Pattern.any().capture("x")
results = parser.find().match_results(Pattern.gelu(x))

for i, r in enumerate(results):
    dot = r.as_query().to_dot()
    with open(f"gelu_{i}.dot", "w") as f:
        f.write(dot)
```

---

## 8. Exporting Subgraphs

### Export the backbone (everything up to the classifier)

```python
classifier = parser.find().find_by_op_type("Gemm")
backbone   = classifier.ancestors()
backbone.to_onnx("backbone.onnx")
```

### Export a matched subgraph for debugging

```python
x       = Pattern.any().capture("x")
results = parser.find().match_results(Pattern.gelu(x))

if results:
    r = results[0]
    r.as_query().to_onnx("gelu_subgraph.onnx")
```

---

## 9. Graph Rewriting

### Remove all Dropout nodes

```python
rw = parser.rewriter()
rw.delete(parser.find().find_by_op_type("Dropout").nodes)
rw.build("no_dropout.onnx")
```

### Fuse GELU subgraphs into a native Gelu op

```python
x  = Pattern.any().capture("x")
rw = parser.rewriter()

for r in parser.find().match_results(Pattern.gelu(x)):
    rw.replace_from_result(r, new_op="Gelu")

rw.build("gelu_fused.onnx")
print(f"Fused {len(parser.find().match_results(Pattern.gelu(x)))} GELU subgraphs")
```

### Fuse SiLU subgraphs

```python
x  = Pattern.any().capture("x")
rw = parser.rewriter()

for r in parser.find().match_results(Pattern.silu(x)):
    rw.replace(
        nodes   = r.nodes,
        new_op  = "HardSwish",          # or custom op
        inputs  = [r.bindings["x"].output[0]],
        outputs = [r.start.output[0]],
    )

rw.build("silu_fused.onnx")
```

### Fuse multiple pattern types in one pass

```python
x  = Pattern.any().capture("x")
rw = parser.rewriter()

gelu_results = parser.find().match_results(Pattern.gelu(x))
silu_results = parser.find().match_results(Pattern.silu(x))

for r in gelu_results:
    rw.replace_from_result(r, new_op="Gelu")

for r in silu_results:
    rw.replace_from_result(r, new_op="Silu")

rw.build("activations_fused.onnx")
```

### Insert a quantisation node before all large Conv layers

```python
rw = parser.rewriter()

def is_large(node, params):
    w = params.get("weight")
    return w is not None and w.shape[1] >= 64

large_convs = []
parser.find().find_by_op_type("Conv").apply(
    lambda n, p: large_convs.append(n) if is_large(n, p) else None
)

for conv in large_convs:
    in_tensor  = conv.input[0]
    quant_out  = f"{in_tensor}_quantised"
    rw.insert_before(
        target_node = conv,
        new_op      = "QuantizeLinear",
        inputs      = [in_tensor],
        outputs     = [quant_out],
    )

rw.build("pre_quantised.onnx")
```

### Stage and conditionally discard edits

```python
rw = parser.rewriter()

x       = Pattern.any().capture("x")
results = parser.find().match_results(Pattern.gelu(x))

if len(results) == 0:
    print("No GELU found — nothing to fuse")
    rw.reset()
else:
    for r in results:
        rw.replace_from_result(r, new_op="Gelu")
    rw.build("fused.onnx")
```

---

## 10. Topological Ordering

### Re-sort an unordered graph

```python
sorted_query = parser.find().topological_sort()
sorted_query.to_onnx("sorted.onnx")
```

### Verify ordering

```python
if parser.find().is_topologically_sorted():
    print("Graph is already in topological order")
else:
    print("Graph nodes are out of topological order")
```

---

## 11. Complete Pipeline: Analysis to Rewrite

This end-to-end example loads a model, analyses its activation functions, prints a report, and fuses them.

```python
from onnx_toolkit import ONNXParser, Pattern

parser = ONNXParser("model.onnx")
print(parser.summary())
print()

x = Pattern.any().capture("x")

activations = {
    "GELU":    Pattern.gelu(x),
    "GeLU-tanh": Pattern.gelu_tanh(x),
    "SiLU":    Pattern.silu(x),
    "Mish":    Pattern.mish(x),
}

fusion_map = {
    "GELU":    "Gelu",
    "GeLU-tanh": "Gelu",
    "SiLU":    "Silu",
    "Mish":    "Mish",
}

rw = parser.rewriter()
total = 0

for act_name, pat in activations.items():
    results = parser.find().match_results(pat)
    if not results:
        continue
    print(f"  {act_name}: {len(results)} occurrence(s)")
    for r in results:
        print(f"    {r.start.name} (input: {r.bindings['x'].name})")
        if act_name in fusion_map:
            rw.replace_from_result(r, new_op=fusion_map[act_name])
            total += 1

print(f"\nFusing {total} activation subgraph(s)...")
if total > 0:
    new_model = rw.build("model_fused.onnx")
    print(f"Saved: {len(new_model.graph.node)} nodes "
          f"(was {len(parser.nodes)})")
else:
    print("Nothing to fuse.")
```
