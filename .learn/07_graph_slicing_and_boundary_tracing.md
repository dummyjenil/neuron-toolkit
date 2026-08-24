# Module 07: Graph Slicing & Boundary Tracing

This module explains how `neuron-toolkit` extracts arbitrary subgraphs from complex neural networks using NetworkX connectivity algorithms (`trace_subgraph_boundaries`).

---

## 1. What is Subgraph Slicing?

Subgraph slicing is the process of taking a large neural network (e.g. ResNet50 or LLaMA) and extracting a small, standalone model containing only the layers between specified `start_points` and `end_points`.

```
[Full Model]:  Input -> Conv1 -> Conv2 -> Layer3 -> Layer4 -> Softmax -> Output
                              └─── Sliced Subgraph ───┘
                                   (Conv2 -> Layer3)
```

---

## 2. The Boundary Tracing Algorithm (`src/neuron_toolkit/_utils.py`)

The heart of slicing is `trace_subgraph_boundaries()`. It uses Graph Theory to determine which operators and tensors belong inside the slice.

```python
def trace_subgraph_boundaries(
    nodes: Sequence[object],
    tensor_map: Mapping[str, object],
    start_points: list[str | object] | str | object,
    end_points: list[str | object] | str | object,
    original_outputs: set[str] | None = None,
) -> tuple[list[object], list[str], list[str]]:
```

### Mathematical & Algorithmic Steps:

#### Step 1: Bipartite Graph Construction

Construct a NetworkX directed graph containing two node types:

- **Operator Nodes (`type="op"`)**: Index $0, 1, \dots, N-1$.
- **Tensor Nodes (`type="tensor"`)**: Wire name strings.

Edges are added from input tensors to operators, and from operators to output tensors:
$$\text{Tensor}_{\text{in}} \longrightarrow \text{Op} \longrightarrow \text{Tensor}_{\text{out}}$$

#### Step 2: Set Resolution ($S$ and $E$)

Resolve start points and end points to node IDs or tensor names in the graph:

- $S$: Set of resolved start nodes/tensors.
- $E$: Set of resolved end nodes/tensors.

#### Step 3: Ancestor & Descendant Intersection

Find all nodes that lie on any path from $S$ to $E$:
$$\text{Descendants}(S) = \{ v \in V \mid S \rightsquigarrow v \}$$
$$\text{Ancestors}(E) = \{ v \in V \mid v \rightsquigarrow E \}$$
$$\text{SubgraphNodes} = \text{Descendants}(S) \cap \text{Ancestors}(E)$$

All operators whose IDs belong to $\text{SubgraphNodes}$ are kept (`kept_ops`).

#### Step 4: Boundary Input Identification

A tensor $t$ is a **boundary input** if:

1. It is consumed by an operator in `kept_ops`.
2. It is NOT produced by any operator in `kept_ops`.
3. It is NOT a constant weight in `tensor_map`.

#### Step 5: Boundary Output Identification

A tensor $t$ is a **boundary output** if:

1. $t \in E$ (explicitly requested as an end point), OR
2. The producer of $t$ is in $E$, OR
3. $t$ has consumers outside `kept_ops` in the original model, OR
4. $t$ was an original graph output of the un-sliced model.

---

## 6. ONNX Slicing Implementation (`backends/onnx/parser.py`)

ONNX slicing uses `trace_subgraph_boundaries()` to get boundary input/output names, then delegates extraction to `onnx.utils.extract_model()`:

```python
_, inputs, outputs = trace_subgraph_boundaries(
    self.nodes, self.tensor_map, start_points, end_points, self.graph_outputs
)

onnx.utils.extract_model(
    input_path=in_path,
    output_path=out_path,
    input_names=inputs,
    output_names=outputs,
    check_model=True,
    infer_shapes=True,
)
```

---

## 7. TFLite Slicing Implementation (`backends/tflite/parser.py`)

Since TFLite lacks a built-in slicing utility, `TFLiteParser.slice()` performs manual FlatBuffer reconstruction:

1. Traces boundaries using `trace_subgraph_boundaries()`.
2. Collects required tensors and re-indexes them.
3. Filters used operator codes.
4. Serializes minimal `SubGraph`, `OperatorCodes`, `Buffers`, and `Model` FlatBuffer structures.

---

## 8. Usage Example

```python
# Extract Layer 2 to Layer 4 as a standalone model:
sliced_graph = graph.slice(
    start_points="layer2_conv1_out",
    end_points="layer4_out",
    output_path="subgraph_layer2_4.onnx"
)
print(f"Original nodes: {len(graph.nodes)}, Sliced nodes: {len(sliced_graph.nodes)}")
```
