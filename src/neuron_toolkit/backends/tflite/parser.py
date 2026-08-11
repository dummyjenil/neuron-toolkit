"""TFLite backend parser."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import ItemsView, Iterator, KeysView
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from neuron_toolkit.backends.base import BaseParser, BaseRewriter
from neuron_toolkit.query import NeuronQuery

if TYPE_CHECKING:
    import numpy as np

    from neuron_toolkit.pattern import MatchResult, Pattern

log = logging.getLogger("neuron_toolkit.backends.tflite")


class LazyTensorMap(dict[str, "np.ndarray"]):
    """Lazy-loading map for TFLite tensors."""

    def __init__(self, model: object, subgraph: object) -> None:
        super().__init__()
        self._model = model
        self._subgraph = subgraph
        self._cache: dict[str, np.ndarray] = {}
        self._name_to_idx: dict[str, int] = {}

        subgraph_tensors_len = cast(Any, subgraph).TensorsLength()
        for i in range(subgraph_tensors_len):
            tensor = cast(Any, subgraph).Tensors(i)
            buffer_idx = tensor.Buffer()
            if buffer_idx > 0:
                buffer = cast(Any, model).Buffers(buffer_idx)
                if (
                    buffer is not None
                    and not buffer.DataIsNone()
                    and buffer.DataLength() > 0
                ):
                    name = tensor.Name().decode("utf-8")
                    self._name_to_idx[name] = i

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._cache:
            return self._cache[key]
        if key in self._name_to_idx:
            idx = self._name_to_idx[key]
            tensor = cast(Any, self._subgraph).Tensors(idx)
            buffer_idx = tensor.Buffer()
            if buffer_idx > 0:
                buffer = cast(Any, self._model).Buffers(buffer_idx)
                data = buffer.DataAsNumpy()
                if data is not None and len(data) > 0:
                    import numpy as np

                    from neuron_toolkit.backends.tflite.utils import _TFLITE_DTYPE_TO_NP

                    dtype_code = tensor.Type()
                    dtype_str = _TFLITE_DTYPE_TO_NP.get(dtype_code, "float32")
                    arr = np.frombuffer(data, dtype=dtype_str)

                    shape = [tensor.Shape(j) for j in range(tensor.ShapeLength())]
                    arr = arr.reshape(shape)

                    self._cache[key] = arr
                    return arr
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        return (
            key in self._name_to_idx or key in self._cache or super().__contains__(key)
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get item by key with optional default."""
        try:
            return self[key]
        except KeyError:
            return default

    def __len__(self) -> int:
        return len(self._name_to_idx)

    def __iter__(self) -> Iterator[str]:
        return iter(self._name_to_idx)

    def keys(self) -> KeysView[str]:
        return self._name_to_idx.keys()

    def items(self) -> ItemsView[str, np.ndarray]:
        return {k: self[k] for k in self._name_to_idx}.items()


@dataclass
class TFLiteNode:
    """Mock-up of a TFLite node that looks like an ONNX node for the query engine."""

    op_type: str
    name: str
    input: list[str]
    output: list[str]
    attrs: dict[str, object] = field(default_factory=dict)
    op_idx: int | None = None


class TFLiteParser(BaseParser):
    """Load a TFLite model and expose it for querying and pattern matching."""

    def __init__(self, tflite_source: str | bytes | object, **_kwargs: object) -> None:
        """Initialize the TFLiteParser."""
        if isinstance(tflite_source, str):
            log.info("Loading TFLite model from path: %r", tflite_source)
            self.path: str | None = tflite_source
        else:
            log.info("Loading TFLite model from bytes or object")
            self.path = None

        self._source = tflite_source
        self.nodes: list[TFLiteNode] = []
        self.tensor_map: LazyTensorMap | dict[str, np.ndarray] = {}
        self.graph_inputs: set[str] = set()
        self.graph_outputs: set[str] = set()
        self.shape_info: dict[str, Any] = {}
        self.quantization_info: dict[str, dict[str, Any]] = {}
        self.sparsity_info: dict[str, dict[str, Any]] = {}

        self._load_model()

    def _load_model(self) -> None:
        """Parse TFLite flatbuffer using the 'tflite' package."""
        try:
            import tflite
        except ImportError as exc:
            log.exception("TFLite package not found. Please install 'tflite'.")
            msg = "TFLite package not found."
            raise RuntimeError(msg) from exc

        from neuron_toolkit.backends.tflite.utils import (
            _build_shape_info,
            _get_tflite_attr,
        )

        model: Any
        if isinstance(self._source, str):
            from pathlib import Path

            buf = Path(self._source).read_bytes()
            model = tflite.Model.GetRootAsModel(buf, 0)
        elif isinstance(self._source, bytes):
            model = tflite.Model.GetRootAsModel(self._source, 0)
        else:
            # Assume already parsed tflite.Model
            model = cast(Any, self._source)

        # TFLite can have multiple subgraphs, we take the primary one (index 0)
        subgraph = model.Subgraphs(0)
        self._model = model
        self._subgraph = subgraph

        # Build lazy tensor map
        self.tensor_map = LazyTensorMap(model, subgraph)

        # Pre-decode tensor names once to avoid repeated FlatBuffer indexing & UTF-8 decoding
        tensor_names = [
            subgraph.Tensors(i).Name().decode("utf-8")
            for i in range(subgraph.TensorsLength())
        ]
        self._tensor_to_idx = {name: i for i, name in enumerate(tensor_names)}

        # Build shape info
        self.shape_info = _build_shape_info(
            model, subgraph, self.quantization_info, self.sparsity_info
        )

        # Pre-build builtin operator code map for O(1) opcode resolution
        builtin_op_map = {
            v: k
            for k, v in tflite.BuiltinOperator.__dict__.items()
            if isinstance(v, int) and not k.startswith("__")
        }

        # Build nodes
        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)
            opcode_idx = op.OpcodeIndex()
            opcode = model.OperatorCodes(opcode_idx)

            # Get OpType
            builtin_code = opcode.BuiltinCode()

            if builtin_code != tflite.BuiltinOperator.CUSTOM:
                op_type = builtin_op_map.get(builtin_code, "UNKNOWN")
            else:
                op_type = opcode.CustomCode().decode("utf-8")

            # Inputs and Outputs
            inputs = []
            for j in range(op.InputsLength()):
                t_idx = op.Inputs(j)
                if t_idx != -1 and t_idx < len(tensor_names):
                    inputs.append(tensor_names[t_idx])

            outputs = []
            for j in range(op.OutputsLength()):
                t_idx = op.Outputs(j)
                if t_idx != -1 and t_idx < len(tensor_names):
                    outputs.append(tensor_names[t_idx])

            # Attributes
            attrs = _get_tflite_attr(op, op_type)

            self.nodes.append(
                TFLiteNode(
                    op_type=op_type,
                    name=f"{op_type}_{i}",
                    input=inputs,
                    output=outputs,
                    attrs=attrs,
                    op_idx=i,
                )
            )

        # Graph inputs/outputs
        for i in range(subgraph.InputsLength()):
            idx = subgraph.Inputs(i)
            if idx < len(tensor_names):
                self.graph_inputs.add(tensor_names[idx])
        for i in range(subgraph.OutputsLength()):
            idx = subgraph.Outputs(i)
            if idx < len(tensor_names):
                self.graph_outputs.add(tensor_names[idx])

        log.info(
            "Model loaded: %d nodes, %d tensors, %d inputs, %d outputs",
            len(self.nodes),
            len(self.tensor_map),
            len(self.graph_inputs),
            len(self.graph_outputs),
        )

    def find(self) -> NeuronQuery:
        """Return a NeuronQuery over all nodes."""
        return NeuronQuery(
            self.nodes,
            self.tensor_map,
            self.nodes,
            self.graph_inputs,
            self.graph_outputs,
            self.shape_info,
            backend=self,
        )

    def pattern_detect(
        self,
        pattern: Pattern,
        start_node: object | None = None,
        end_node: object | None = None,
    ) -> MatchResult | None:
        """Create a PatternDetector bound to this model and call match()."""
        from neuron_toolkit._utils import _GraphShim
        from neuron_toolkit.pattern import PatternDetector

        shim = _GraphShim(self.nodes, self.tensor_map, self.shape_info, backend=self)
        det = PatternDetector(shim, start_node=start_node, end_node=end_node)
        return det.match(pattern)

    def get_node_attrs(self, node: object) -> dict[str, object]:
        """Extract attributes from a TFLite node."""
        return getattr(node, "attrs", {})

    def is_constant_node(self, node: object) -> bool:
        """TFLite doesn't typically use explicit Constant nodes in the same way."""
        return getattr(node, "op_type", None) == "CONST"

    def get_constant_value(self, node: object) -> object | None:
        """Extract value from TFLite CONST node if available."""
        return getattr(node, "attrs", {}).get("value", None)

    def summary(self) -> str:
        """Return a summary of the model."""
        op_counts = Counter(n.op_type for n in self.nodes)
        lines = [
            "TFLite model summary",
            f"  Nodes        : {len(self.nodes)}",
            f"  Tensors      : {len(self.shape_info)}",
            f"  Graph inputs : {len(self.graph_inputs)}",
            f"  Graph outputs: {len(self.graph_outputs)}",
            f"  Op types ({len(op_counts)}):",
        ]
        for op, cnt in op_counts.most_common():
            lines.append(f"    {op:<24} {cnt:>5}")
        return "\n".join(lines)

    def rewriter(self) -> BaseRewriter:
        """Return a rewriter for TFLite models."""
        from neuron_toolkit.backends.tflite.rewriter import (
            TFLiteRewriter,
        )

        return TFLiteRewriter(self)

    @property
    def source(self) -> object:
        """Return the original source."""
        return self._source

    def slice(
        self,
        start_points: list[str | object] | str | object,
        end_points: list[str | object] | str | object,
        output_path: str | None = None,
    ) -> TFLiteParser:
        """Slice the TFLite model between start and end points and return a new TFLiteParser."""
        from pathlib import Path

        import flatbuffers
        import tflite

        from neuron_toolkit._utils import trace_subgraph_boundaries
        from neuron_toolkit.backends.tflite.rewriter import (
            _copy_flatbuffer_table,
            _copy_quantization,
            _copy_sparsity,
        )

        vtable_cache = {}

        # 1. Trace the boundary nodes/tensors
        kept_ops, new_inputs, new_outputs = trace_subgraph_boundaries(
            self.nodes, self.tensor_map, start_points, end_points, self.graph_outputs
        )

        # 2. Gather all tensors that are consumed/produced by kept_ops, or are boundary inputs/outputs
        kept_tensor_names = set()
        for op in kept_ops:
            for inp in op.input:
                if inp:
                    kept_tensor_names.add(inp)
            for out in op.output:
                if out:
                    kept_tensor_names.add(out)
        for t in new_inputs:
            kept_tensor_names.add(t)
        for t in new_outputs:
            kept_tensor_names.add(t)

        kept_tensor_names = sorted(kept_tensor_names)
        tensor_name_to_new_idx = {
            name: idx for idx, name in enumerate(kept_tensor_names)
        }

        # 3. Gather unique operator codes used by kept operators
        used_op_types = sorted({op.op_type for op in kept_ops})
        opcode_map = {op_type: idx for idx, op_type in enumerate(used_op_types)}

        # 4. Map buffers (initializers) for the kept tensors
        new_buffers = [b""]  # Index 0 is always empty
        tensor_to_buffer_idx = {}
        for t_name in kept_tensor_names:
            orig_idx = self._tensor_to_idx.get(t_name)
            if orig_idx is not None:
                t = self._subgraph.Tensors(orig_idx)
                buf_idx = t.Buffer()
                if buf_idx > 0:
                    buf = self._model.Buffers(buf_idx)
                    if (
                        buf is not None
                        and not buf.DataIsNone()
                        and buf.DataLength() > 0
                    ):
                        data = bytes(buf.DataAsNumpy())
                        new_buffers.append(data)
                        tensor_to_buffer_idx[t_name] = len(new_buffers) - 1
                        continue
            tensor_to_buffer_idx[t_name] = 0

        # 5. Build Flatbuffer
        builder = flatbuffers.Builder(1024 * 1024)

        # A. Serialize Tensors
        tensor_offsets = []
        for t_name in kept_tensor_names:
            orig_idx = self._tensor_to_idx.get(t_name)
            name_offset = builder.CreateString(t_name)

            if orig_idx is not None:
                t = self._subgraph.Tensors(orig_idx)
                shape = [t.Shape(j) for j in range(t.ShapeLength())]
                dtype_code = t.Type()
                buf_idx = tensor_to_buffer_idx[t_name]
                quant_offset = _copy_quantization(builder, t.Quantization())
                sparsity_offset = _copy_sparsity(builder, t.Sparsity())
                is_var = t.IsVariable()
                has_rank = t.HasRank()

                shape_sig = (
                    [t.ShapeSignature(j) for j in range(t.ShapeSignatureLength())]
                    if not t.ShapeSignatureIsNone()
                    else []
                )
                shape_sig_vec = 0
                if shape_sig:
                    tflite.TensorStartShapeSignatureVector(builder, len(shape_sig))
                    for s in reversed(shape_sig):
                        builder.PrependInt32(s)
                    shape_sig_vec = builder.EndVector()
            else:
                shape = []
                dtype_code = 0
                buf_idx = 0
                quant_offset = 0
                sparsity_offset = 0
                is_var = False
                has_rank = False
                shape_sig_vec = 0

            tflite.TensorStartShapeVector(builder, len(shape))
            for s in reversed(shape):
                builder.PrependInt32(s)
            shape_vec = builder.EndVector()

            tflite.TensorStart(builder)
            tflite.TensorAddShape(builder, shape_vec)
            tflite.TensorAddType(builder, dtype_code)
            tflite.TensorAddBuffer(builder, buf_idx)
            tflite.TensorAddName(builder, name_offset)
            if quant_offset:
                tflite.TensorAddQuantization(builder, quant_offset)
            if sparsity_offset:
                tflite.TensorAddSparsity(builder, sparsity_offset)
            tflite.TensorAddIsVariable(builder, is_var)
            tflite.TensorAddHasRank(builder, has_rank)
            if shape_sig_vec:
                tflite.TensorAddShapeSignature(builder, shape_sig_vec)

            tensor_offsets.append(tflite.TensorEnd(builder))

        tflite.SubGraphStartTensorsVector(builder, len(tensor_offsets))
        for t in reversed(tensor_offsets):
            builder.PrependUOffsetTRelative(t)
        tensors_vec = builder.EndVector()

        # B. Serialize Operators (topologically sorted)
        import networkx as nx

        op_dep_graph = nx.DiGraph()
        for idx, op in enumerate(kept_ops):
            op_dep_graph.add_node(idx, op=op)

        for idx_a, op_a in enumerate(kept_ops):
            for out in op_a.output:
                if not out:
                    continue
                for idx_b, op_b in enumerate(kept_ops):
                    if idx_a == idx_b:
                        continue
                    if out in op_b.input:
                        op_dep_graph.add_edge(idx_a, idx_b)

        try:
            sorted_indices = list(nx.topological_sort(op_dep_graph))
            sorted_kept_ops = [kept_ops[idx] for idx in sorted_indices]
        except nx.NetworkXCyclicError:
            sorted_kept_ops = kept_ops

        op_offsets = []
        for op_node in sorted_kept_ops:
            orig_op = self._subgraph.Operators(op_node.op_idx)

            mapped_inputs = [
                tensor_name_to_new_idx[inp]
                for inp in op_node.input
                if inp in tensor_name_to_new_idx
            ]
            mapped_outputs = [
                tensor_name_to_new_idx[out]
                for out in op_node.output
                if out in tensor_name_to_new_idx
            ]

            tflite.OperatorStartInputsVector(builder, len(mapped_inputs))
            for x in reversed(mapped_inputs):
                builder.PrependInt32(x)
            inputs_vec = builder.EndVector()

            tflite.OperatorStartOutputsVector(builder, len(mapped_outputs))
            for x in reversed(mapped_outputs):
                builder.PrependInt32(x)
            outputs_vec = builder.EndVector()

            opts = orig_op.BuiltinOptions()
            opt_offset = (
                _copy_flatbuffer_table(builder, opts, vtable_cache) if opts else 0
            )

            tflite.OperatorStart(builder)
            tflite.OperatorAddOpcodeIndex(builder, opcode_map[op_node.op_type])
            tflite.OperatorAddInputs(builder, inputs_vec)
            tflite.OperatorAddOutputs(builder, outputs_vec)
            if orig_op.BuiltinOptionsType():
                tflite.OperatorAddBuiltinOptionsType(
                    builder, orig_op.BuiltinOptionsType()
                )
            if opt_offset:
                tflite.OperatorAddBuiltinOptions(builder, opt_offset)
            op_offsets.append(tflite.OperatorEnd(builder))

        tflite.SubGraphStartOperatorsVector(builder, len(op_offsets))
        for o in reversed(op_offsets):
            builder.PrependUOffsetTRelative(o)
        ops_vec = builder.EndVector()

        # C. Inputs and Outputs for the SubGraph
        sub_inputs = [tensor_name_to_new_idx[inp] for inp in new_inputs]
        tflite.SubGraphStartInputsVector(builder, len(sub_inputs))
        for i in reversed(sub_inputs):
            builder.PrependInt32(i)
        sub_inputs_vec = builder.EndVector()

        sub_outputs = [tensor_name_to_new_idx[out] for out in new_outputs]
        tflite.SubGraphStartOutputsVector(builder, len(sub_outputs))
        for o in reversed(sub_outputs):
            builder.PrependInt32(o)
        sub_outputs_vec = builder.EndVector()

        # D. Build SubGraph
        subgraph_name = builder.CreateString("sliced_subgraph")
        tflite.SubGraphStart(builder)
        tflite.SubGraphAddTensors(builder, tensors_vec)
        tflite.SubGraphAddInputs(builder, sub_inputs_vec)
        tflite.SubGraphAddOutputs(builder, sub_outputs_vec)
        tflite.SubGraphAddOperators(builder, ops_vec)
        tflite.SubGraphAddName(builder, subgraph_name)
        subgraph_offset = tflite.SubGraphEnd(builder)

        tflite.ModelStartSubgraphsVector(builder, 1)
        builder.PrependUOffsetTRelative(subgraph_offset)
        subgraphs_vec = builder.EndVector()

        # E. Rebuild OperatorCodes
        opcode_offsets = []
        sorted_opcodes = sorted(opcode_map.items(), key=lambda x: x[1])
        for name, _ in sorted_opcodes:
            if hasattr(tflite.BuiltinOperator, name) and not name.startswith("__"):
                builtin_code = getattr(tflite.BuiltinOperator, name)
            else:
                builtin_code = tflite.BuiltinOperator.CUSTOM

            custom_code_offset = 0
            if builtin_code == tflite.BuiltinOperator.CUSTOM:
                custom_code_offset = builder.CreateString(name)

            tflite.OperatorCodeStart(builder)
            deprecated_builtin_max = 127
            if builtin_code < deprecated_builtin_max:
                tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, builtin_code)
            tflite.OperatorCodeAddBuiltinCode(builder, builtin_code)
            if custom_code_offset:
                tflite.OperatorCodeAddCustomCode(builder, custom_code_offset)
            opcode_offsets.append(tflite.OperatorCodeEnd(builder))

        tflite.ModelStartOperatorCodesVector(builder, len(opcode_offsets))
        for o in reversed(opcode_offsets):
            builder.PrependUOffsetTRelative(o)
        opcodes_vec = builder.EndVector()

        # F. Rebuild Buffers
        buffer_offsets = []
        for buf_data in new_buffers:
            data_vec = builder.CreateByteVector(buf_data) if buf_data else 0

            tflite.BufferStart(builder)
            if data_vec:
                tflite.BufferAddData(builder, data_vec)
            buffer_offsets.append(tflite.BufferEnd(builder))

        tflite.ModelStartBuffersVector(builder, len(buffer_offsets))
        for b in reversed(buffer_offsets):
            builder.PrependUOffsetTRelative(b)
        buffers_vec = builder.EndVector()

        # G. Rebuild Model
        desc = builder.CreateString("sliced by neuron_toolkit")
        tflite.ModelStart(builder)
        tflite.ModelAddVersion(builder, self._model.Version())
        tflite.ModelAddOperatorCodes(builder, opcodes_vec)
        tflite.ModelAddSubgraphs(builder, subgraphs_vec)
        tflite.ModelAddBuffers(builder, buffers_vec)
        tflite.ModelAddDescription(builder, desc)
        model_offset = tflite.ModelEnd(builder)

        builder.Finish(model_offset, file_identifier=b"TFL3")
        output_data = bytes(builder.Output())

        if output_path:
            Path(output_path).write_bytes(output_data)

        return TFLiteParser(output_data)
