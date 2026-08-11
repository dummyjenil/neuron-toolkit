# Module 06: Staged Graph Rewriting (`NeuronRewriter`)

This module explains how graph transformations (replacing, deleting, or inserting nodes) are safely staged and executed across ONNX and TFLite backends.

---

## 1. The Staging Pattern Philosophy

Direct inline mutation of computational graphs is dangerous. Deleting or inserting nodes directly while iterating over a graph can break topological ordering, invalidate index pointers, and cause memory corruption.

`neuron-toolkit` uses a **Staged Graph Rewriter**:
1. You register intended changes (`replace`, `delete`, `insert_before`, `register_initializer`).
2. Changes are stored in temporary staging lists (`_to_remove_ids`, `_to_insert`).
3. When `build()` is called, all edits are applied in a single pass, followed by **topological sorting** and **shape re-inference**.

---

## 2. Base Abstract Rewriter (`src/neuron_toolkit/backends/base.py`)

```python
class BaseRewriter(ABC):
    @abstractmethod
    def replace(self, nodes: Sequence[object], new_op: str, inputs: list[str], outputs: list[str], ...) -> BaseRewriter: ...

    @abstractmethod
    def replace_from_result(self, result: MatchResult, new_op: str, ...) -> BaseRewriter: ...

    @abstractmethod
    def delete(self, nodes: Sequence[object]) -> BaseRewriter: ...

    @abstractmethod
    def insert_before(self, target_node: object, new_op: str, ...) -> BaseRewriter: ...

    @abstractmethod
    def reset(self) -> BaseRewriter: ...

    @abstractmethod
    def build(self, output_path: str | None = None) -> object: ...
```

---

## 3. ONNX Rewriter (`src/neuron_toolkit/backends/onnx/rewriter.py`)

### Staging Edits:
```python
def replace(self, nodes, new_op, inputs, outputs, name=None, **attrs):
    for n in nodes:
        self._to_remove_ids.add(id(n)) # Stage for deletion by object ID
    
    new_node = helper.make_node(new_op, inputs=inputs, outputs=outputs, name=name, **attrs)
    self._to_insert.append(new_node) # Stage new node for insertion
    return self
```

### Applying Edits in `build()`:
1. **Node Filtering**:
   ```python
   kept_nodes = [n for n in orig_graph.node if id(n) not in self._to_remove_ids]
   all_nodes = kept_nodes + self._to_insert
   ```

2. **Topological Sorting via NetworkX**:
   New inserted nodes might be appended at the end of the list out of order. `ONNXRewriter` constructs a temporary `DiGraph` linking node producers to consumers and runs `nx.topological_sort(g)`:
   ```python
   sorted_node_ids = list(nx.topological_sort(g))
   final_nodes = [g.nodes[nid]["proto"] for nid in sorted_node_ids]
   ```

3. **Graph Reconstruction & Shape Inference**:
   Creates a new `ModelProto` with the updated node list and registered initializers, then executes `onnx.shape_inference.infer_shapes(new_model)`.

---

## 4. TFLite Rewriter (`src/neuron_toolkit/backends/tflite/rewriter.py`)

Rewriting a TFLite FlatBuffer binary is significantly more complex because FlatBuffers are immutable byte arrays built bottom-up.

### FlatBuffer Builder Pipeline:
1. **Operator Codes Registration (`_prepare_opcodes`)**: Collects existing opcodes and assigns new integer codes for inserted ops.
2. **Tensor Table Serialization (`_build_tensors`)**: Re-serializes existing tensors and appends new registered weight tensors (`register_tensor`).
3. **FlatBuffer Options Serialization**:
   TFLite requires options tables to be pre-built in the FlatBuffer builder. `_OP_TYPE_TO_BUILTIN_OPTIONS` converts dict attributes to FlatBuffer option offsets:
   ```python
   def _serialize_AddOptions(builder, attrs):
       act_enum = getattr(tflite.ActivationFunctionType, attrs.get("fused_activation_function", "NONE"))
       tflite.AddOptionsStart(builder)
       tflite.AddOptionsAddFusedActivationFunction(builder, act_enum)
       return tflite.AddOptionsEnd(builder)
   ```
4. **Topological Sorting**: Like ONNX, TFLite operators are topologically sorted using NetworkX before serialization.
5. **FlatBuffer Table Cloning (`_copy_flatbuffer_table`)**: Unmodified options tables are copied byte-for-byte using FlatBuffer vtable offsets:
   ```python
   vtable_offset = flatbuffers.encode.Get(int32, bytes_data, pos)
   ```

---

## 5. Convenient Fluent Shortcuts

### `graph.replace(pattern, new_op, **attrs)`
Instead of manually matching and rewriting, you can call `.replace()` directly on `NeuronGraph`:

```python
# Replace all Relu nodes with LeakyRelu in one line:
graph.replace(Pattern.op("Relu"), "LeakyRelu", alpha=0.1).build("modified.onnx")
```
