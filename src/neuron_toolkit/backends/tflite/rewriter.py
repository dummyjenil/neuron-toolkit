"""TFLite backend rewriter."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import networkx as nx

from neuron_toolkit.backends.base import BaseRewriter

if TYPE_CHECKING:
    from neuron_toolkit.backends.tflite.parser import TFLiteParser
    from neuron_toolkit.pattern import MatchResult

try:
    import flatbuffers
    import tflite

    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False

log = logging.getLogger("neuron_toolkit.backends.tflite.rewriter")

# Numpy dtype string to TFLite tensor type integer code mapping
_NP_DTYPE_TO_TFLITE: dict[str | int, int] = {
    "float32": 0,
    "float16": 1,
    "int32": 2,
    "uint8": 3,
    "int64": 4,
    "bool": 6,
    "int16": 7,
    "int8": 9,
    "float64": 10,
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    6: 6,
    7: 7,
    9: 9,
    10: 10,
}


def _copy_quantization(builder: flatbuffers.Builder, q: Any) -> int:
    """Serialize a copy of QuantizationParameters to the flatbuffer builder."""
    if q is None:
        return 0
    try:
        min_vals = [q.Min(j) for j in range(q.MinLength())] if not q.MinIsNone() else []
        max_vals = [q.Max(j) for j in range(q.MaxLength())] if not q.MaxIsNone() else []
        scales = (
            [q.Scale(j) for j in range(q.ScaleLength())] if not q.ScaleIsNone() else []
        )
        zero_points = (
            [q.ZeroPoint(j) for j in range(q.ZeroPointLength())]
            if not q.ZeroPointIsNone()
            else []
        )
    except (AttributeError, TypeError):
        return 0

    min_vec = 0
    if min_vals:
        tflite.QuantizationParametersStartMinVector(builder, len(min_vals))
        for val in reversed(min_vals):
            builder.PrependFloat32(val)
        min_vec = builder.EndVector()

    max_vec = 0
    if max_vals:
        tflite.QuantizationParametersStartMaxVector(builder, len(max_vals))
        for val in reversed(max_vals):
            builder.PrependFloat32(val)
        max_vec = builder.EndVector()

    scale_vec = 0
    if scales:
        tflite.QuantizationParametersStartScaleVector(builder, len(scales))
        for val in reversed(scales):
            builder.PrependFloat32(val)
        scale_vec = builder.EndVector()

    zp_vec = 0
    if zero_points:
        tflite.QuantizationParametersStartZeroPointVector(builder, len(zero_points))
        for val in reversed(zero_points):
            builder.PrependInt64(val)
        zp_vec = builder.EndVector()

    tflite.QuantizationParametersStart(builder)
    if min_vec:
        tflite.QuantizationParametersAddMin(builder, min_vec)
    if max_vec:
        tflite.QuantizationParametersAddMax(builder, max_vec)
    if scale_vec:
        tflite.QuantizationParametersAddScale(builder, scale_vec)
    if zp_vec:
        tflite.QuantizationParametersAddZeroPoint(builder, zp_vec)
    try:
        tflite.QuantizationParametersAddQuantizedDimension(
            builder, q.QuantizedDimension()
        )
    except AttributeError:
        pass
    return tflite.QuantizationParametersEnd(builder)


def _copy_sparsity(builder: flatbuffers.Builder, s: Any) -> int:
    """Serialize a copy of SparsityParameters to the flatbuffer builder."""
    if s is None:
        return 0
    try:
        traversal_order = (
            [s.TraversalOrder(j) for j in range(s.TraversalOrderLength())]
            if not s.TraversalOrderIsNone()
            else []
        )
        block_map = (
            [s.BlockMap(j) for j in range(s.BlockMapLength())]
            if not s.BlockMapIsNone()
            else []
        )
    except (AttributeError, TypeError):
        return 0

    order_vec = 0
    if traversal_order:
        tflite.SparsityParametersStartTraversalOrderVector(
            builder, len(traversal_order)
        )
        for val in reversed(traversal_order):
            builder.PrependInt32(val)
        order_vec = builder.EndVector()

    map_vec = 0
    if block_map:
        tflite.SparsityParametersStartBlockMapVector(builder, len(block_map))
        for val in reversed(block_map):
            builder.PrependInt32(val)
        map_vec = builder.EndVector()

    dim_metadata_offsets = []
    try:
        dim_meta_len = s.DimMetadataLength()
    except (AttributeError, TypeError):
        dim_meta_len = 0

    for i in range(dim_meta_len):
        dm = s.DimMetadata(i)
        if dm is None:
            continue
        try:
            fmt = dm.Format()
            dense_size = dm.DenseSize()
            tflite.DimensionMetadataStart(builder)
            tflite.DimensionMetadataAddFormat(builder, fmt)
            tflite.DimensionMetadataAddDenseSize(builder, dense_size)
            dim_metadata_offsets.append(tflite.DimensionMetadataEnd(builder))
        except AttributeError:
            pass

    dim_metadata_vec = 0
    if dim_metadata_offsets:
        tflite.SparsityParametersStartDimMetadataVector(
            builder, len(dim_metadata_offsets)
        )
        for off in reversed(dim_metadata_offsets):
            builder.PrependUOffsetTRelative(off)
        dim_metadata_vec = builder.EndVector()

    tflite.SparsityParametersStart(builder)
    if order_vec:
        tflite.SparsityParametersAddTraversalOrder(builder, order_vec)
    if map_vec:
        tflite.SparsityParametersAddBlockMap(builder, map_vec)
    if dim_metadata_vec:
        tflite.SparsityParametersAddDimMetadata(builder, dim_metadata_vec)
    return (
        s.SparsityParametersEnd(builder)
        if hasattr(s, "SparsityParametersEnd")
        else tflite.SparsityParametersEnd(builder)
    )


def _copy_signature_def(
    builder: flatbuffers.Builder, sig: Any, tensor_name_to_idx: dict[str, int]
) -> int:
    """Serialize a copy of SignatureDef to the flatbuffer builder, mapping tensor names to new indices."""
    if sig is None:
        return 0

    # Copy Inputs
    input_offsets = []
    try:
        inputs_len = sig.InputsLength()
    except (AttributeError, TypeError):
        inputs_len = 0

    for i in range(inputs_len):
        inp = sig.Inputs(i)
        if inp is None:
            continue
        name_str = inp.Name().decode("utf-8") if inp.Name() else ""
        name_offset = builder.CreateString(name_str)
        new_idx = tensor_name_to_idx.get(name_str, inp.TensorIndex())

        tflite.TensorMapStart(builder)
        tflite.TensorMapAddName(builder, name_offset)
        tflite.TensorMapAddTensorIndex(builder, new_idx)
        input_offsets.append(tflite.TensorMapEnd(builder))

    inputs_vec = 0
    if input_offsets:
        tflite.SignatureDefStartInputsVector(builder, len(input_offsets))
        for off in reversed(input_offsets):
            builder.PrependUOffsetTRelative(off)
        inputs_vec = builder.EndVector()

    # Copy Outputs
    output_offsets = []
    try:
        outputs_len = sig.OutputsLength()
    except (AttributeError, TypeError):
        outputs_len = 0

    for i in range(outputs_len):
        out = sig.Outputs(i)
        if out is None:
            continue
        name_str = out.Name().decode("utf-8") if out.Name() else ""
        name_offset = builder.CreateString(name_str)
        new_idx = tensor_name_to_idx.get(name_str, out.TensorIndex())

        tflite.TensorMapStart(builder)
        tflite.TensorMapAddName(builder, name_offset)
        tflite.TensorMapAddTensorIndex(builder, new_idx)
        output_offsets.append(tflite.TensorMapEnd(builder))

    outputs_vec = 0
    if output_offsets:
        tflite.SignatureDefStartOutputsVector(builder, len(output_offsets))
        for off in reversed(output_offsets):
            builder.PrependUOffsetTRelative(off)
        outputs_vec = builder.EndVector()

    # Key
    try:
        key_str = sig.SignatureKey().decode("utf-8") if sig.SignatureKey() else ""
    except AttributeError:
        key_str = ""
    key_offset = builder.CreateString(key_str) if key_str else 0

    tflite.SignatureDefStart(builder)
    if inputs_vec:
        tflite.SignatureDefAddInputs(builder, inputs_vec)
    if outputs_vec:
        tflite.SignatureDefAddOutputs(builder, outputs_vec)
    if key_offset:
        tflite.SignatureDefAddSignatureKey(builder, key_offset)
    try:
        tflite.SignatureDefAddSubgraphIndex(builder, sig.SubgraphIndex())
    except AttributeError:
        pass
    return tflite.SignatureDefEnd(builder)


def _copy_metadata(builder: flatbuffers.Builder, meta: Any) -> int:
    """Serialize a copy of Metadata to the flatbuffer builder."""
    if meta is None:
        return 0
    try:
        name_str = meta.Name().decode("utf-8") if meta.Name() else ""
    except AttributeError:
        name_str = ""
    name_offset = builder.CreateString(name_str) if name_str else 0

    tflite.MetadataStart(builder)
    if name_offset:
        tflite.MetadataAddName(builder, name_offset)
    try:
        tflite.MetadataAddBuffer(builder, meta.Buffer())
    except AttributeError:
        pass
    return tflite.MetadataEnd(builder)


# Option Serializers
def _serialize_AddOptions(builder: flatbuffers.Builder, attrs: dict[str, Any]) -> int:
    activation = attrs.get("fused_activation_function", "NONE")
    act_enum = getattr(
        tflite.ActivationFunctionType, activation, tflite.ActivationFunctionType.NONE
    )
    tflite.AddOptionsStart(builder)
    tflite.AddOptionsAddFusedActivationFunction(builder, act_enum)
    return tflite.AddOptionsEnd(builder)


def _serialize_SubOptions(builder: flatbuffers.Builder, attrs: dict[str, Any]) -> int:
    activation = attrs.get("fused_activation_function", "NONE")
    act_enum = getattr(
        tflite.ActivationFunctionType, activation, tflite.ActivationFunctionType.NONE
    )
    tflite.SubOptionsStart(builder)
    tflite.SubOptionsAddFusedActivationFunction(builder, act_enum)
    return tflite.SubOptionsEnd(builder)


def _serialize_MulOptions(builder: flatbuffers.Builder, attrs: dict[str, Any]) -> int:
    activation = attrs.get("fused_activation_function", "NONE")
    act_enum = getattr(
        tflite.ActivationFunctionType, activation, tflite.ActivationFunctionType.NONE
    )
    tflite.MulOptionsStart(builder)
    tflite.MulOptionsAddFusedActivationFunction(builder, act_enum)
    return tflite.MulOptionsEnd(builder)


def _serialize_DivOptions(builder: flatbuffers.Builder, attrs: dict[str, Any]) -> int:
    activation = attrs.get("fused_activation_function", "NONE")
    act_enum = getattr(
        tflite.ActivationFunctionType, activation, tflite.ActivationFunctionType.NONE
    )
    tflite.DivOptionsStart(builder)
    tflite.DivOptionsAddFusedActivationFunction(builder, act_enum)
    return tflite.DivOptionsEnd(builder)


def _serialize_SoftmaxOptions(
    builder: flatbuffers.Builder, attrs: dict[str, Any]
) -> int:
    tflite.SoftmaxOptionsStart(builder)
    tflite.SoftmaxOptionsAddBeta(builder, float(attrs.get("beta", 1.0)))
    return tflite.SoftmaxOptionsEnd(builder)


def _serialize_ReshapeOptions(
    builder: flatbuffers.Builder, attrs: dict[str, Any]
) -> int:
    shape = attrs.get("new_shape", [])
    shape_vec = 0
    if shape:
        tflite.ReshapeOptionsStartNewShapeVector(builder, len(shape))
        for s in reversed(shape):
            builder.PrependInt32(s)
        shape_vec = builder.EndVector()
    tflite.ReshapeOptionsStart(builder)
    if shape_vec:
        tflite.ReshapeOptionsAddNewShape(builder, shape_vec)
    return tflite.ReshapeOptionsEnd(builder)


def _serialize_Conv2DOptions(
    builder: flatbuffers.Builder, attrs: dict[str, Any]
) -> int:
    padding = attrs.get("padding", "SAME")
    pad_enum = getattr(tflite.Padding, padding, tflite.Padding.SAME)
    activation = attrs.get("fused_activation_function", "NONE")
    act_enum = getattr(
        tflite.ActivationFunctionType, activation, tflite.ActivationFunctionType.NONE
    )

    tflite.Conv2DOptionsStart(builder)
    tflite.Conv2DOptionsAddPadding(builder, pad_enum)
    tflite.Conv2DOptionsAddStrideW(builder, int(attrs.get("stride_w", 1)))
    tflite.Conv2DOptionsAddStrideH(builder, int(attrs.get("stride_h", 1)))
    tflite.Conv2DOptionsAddFusedActivationFunction(builder, act_enum)
    tflite.Conv2DOptionsAddDilationWFactor(
        builder, int(attrs.get("dilation_w_factor", 1))
    )
    tflite.Conv2DOptionsAddDilationHFactor(
        builder, int(attrs.get("dilation_h_factor", 1))
    )
    return tflite.Conv2DOptionsEnd(builder)


_OP_TYPE_TO_BUILTIN_OPTIONS: dict[str, tuple[int, Any]] = {
    "ADD": (tflite.BuiltinOptions.AddOptions, _serialize_AddOptions),
    "SUB": (tflite.BuiltinOptions.SubOptions, _serialize_SubOptions),
    "MUL": (tflite.BuiltinOptions.MulOptions, _serialize_MulOptions),
    "DIV": (tflite.BuiltinOptions.DivOptions, _serialize_DivOptions),
    "RESHAPE": (tflite.BuiltinOptions.ReshapeOptions, _serialize_ReshapeOptions),
    "SOFTMAX": (tflite.BuiltinOptions.SoftmaxOptions, _serialize_SoftmaxOptions),
    "CONV_2D": (tflite.BuiltinOptions.Conv2DOptions, _serialize_Conv2DOptions),
}


class TFLiteRewriter(BaseRewriter):
    """Transformation engine for TFLite models.

    Uses flatbuffer builders to reconstruct modified model flatbuffers.
    """

    def __init__(self, parser: TFLiteParser) -> None:
        """Initialize with a TFLiteParser instance."""
        self._parser = parser
        self._to_remove: set[str] = set()  # node names to drop
        self._to_insert: list[dict[str, Any]] = []  # new nodes to add
        self._new_tensors: dict[str, dict[str, Any]] = {}  # newly registered tensors
        self._new_buffers: list[bytes] = []  # buffers for newly registered tensors

    def replace(
        self,
        nodes: Sequence[object],
        new_op: str,
        inputs: list[str],
        outputs: list[str],
        name: str | None = None,
        **attrs: object,
    ) -> TFLiteRewriter:
        """Replace *nodes* with a single new node of type *new_op*."""
        for n in nodes:
            name_str = getattr(n, "name", "")
            if name_str:
                self._to_remove.add(name_str)

        node_name = name or f"{new_op}_{len(self._to_insert)}_rewrite"
        self._to_insert.append(
            {
                "op_type": new_op,
                "inputs": inputs,
                "outputs": outputs,
                "name": node_name,
                "attrs": attrs,
            }
        )
        log.info(
            "replace(): scheduled removal of %d node(s), inserting %r",
            len(nodes),
            node_name,
        )
        return self

    def replace_from_result(
        self,
        result: MatchResult,
        new_op: str,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        name: str | None = None,
        **attrs: object,
    ) -> TFLiteRewriter:
        """Convenience wrapper: replace all nodes in *result* with a new op."""
        nodes_list: list[Any] = result.nodes
        start_node: Any = result.start
        internal_outputs = {out for n in nodes_list for out in n.output}
        if inputs is None:
            ins = []
            for n in nodes_list:
                for inp in n.input:
                    if inp and inp not in internal_outputs and inp not in ins:
                        ins.append(inp)
            inputs = ins

        if outputs is None:
            outputs = list(start_node.output)

        return self.replace(result.nodes, new_op, inputs, outputs, name=name, **attrs)

    def delete(self, nodes: Sequence[object]) -> TFLiteRewriter:
        """Remove *nodes* from the graph entirely."""
        for n in nodes:
            name_str = getattr(n, "name", "")
            if name_str:
                self._to_remove.add(name_str)
        log.info("delete(): scheduled removal of %d node(s)", len(nodes))
        return self

    def insert_before(
        self,
        target_node: object,
        new_op: str,
        inputs: list[str],
        outputs: list[str],
        name: str | None = None,
        **attrs: object,
    ) -> TFLiteRewriter:
        """Insert a new node whose outputs feed *target_node*."""
        node_name = name or f"{new_op}_{len(self._to_insert)}_insert"
        self._to_insert.append(
            {
                "op_type": new_op,
                "inputs": inputs,
                "outputs": outputs,
                "name": node_name,
                "attrs": attrs,
            }
        )
        target_name = getattr(target_node, "name", "<unnamed>")
        log.info(
            "insert_before(%r): scheduled insertion of %r",
            target_name,
            node_name,
        )
        return self

    def register_tensor(
        self,
        name: str,
        shape: list[int],
        dtype: int | str,
        buffer_data: bytes | None = None,
    ) -> TFLiteRewriter:
        """Register a new tensor to be added to the rewritten model graph."""
        self._new_tensors[name] = {
            "shape": shape,
            "dtype": dtype,
            "buffer_data": buffer_data,
        }
        return self

    def reset(self) -> TFLiteRewriter:
        """Discard all pending edits."""
        self._to_remove.clear()
        self._to_insert.clear()
        self._new_tensors.clear()
        self._new_buffers.clear()
        return self

    def _prepare_opcodes(self, orig_model: Any) -> dict[str, int]:
        opcode_map: dict[str, int] = {}
        for i in range(orig_model.OperatorCodesLength()):
            oc = orig_model.OperatorCodes(i)
            builtin = oc.BuiltinCode()

            op_name = "UNKNOWN"
            if builtin != tflite.BuiltinOperator.CUSTOM:
                for k, v in tflite.BuiltinOperator.__dict__.items():
                    if isinstance(v, int) and v == builtin and not k.startswith("__"):
                        op_name = k
                        break
            else:
                op_name = oc.CustomCode().decode("utf-8")

            opcode_map[op_name] = i

        for item in self._to_insert:
            op_type = item["op_type"]
            if op_type not in opcode_map:
                opcode_map[op_type] = len(opcode_map)
        return opcode_map

    def _build_tensors(
        self, builder: flatbuffers.Builder, orig_subgraph: Any, orig_model: Any
    ) -> tuple[int, dict[str, int]]:
        tensor_offsets = []
        tensor_name_to_idx = {}

        # 1. Rebuild original tensors
        for i in range(orig_subgraph.TensorsLength()):
            t = orig_subgraph.Tensors(i)
            name_str = t.Name().decode("utf-8") if t.Name() else f"tensor_{i}"
            tensor_name_to_idx[name_str] = i

            name = builder.CreateString(name_str)
            shape = [t.Shape(j) for j in range(t.ShapeLength())]

            tflite.TensorStartShapeVector(builder, len(shape))
            for s in reversed(shape):
                builder.PrependInt32(s)
            shape_vec = builder.EndVector()

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

            quant_offset = _copy_quantization(builder, t.Quantization())
            sparsity_offset = _copy_sparsity(builder, t.Sparsity())

            tflite.TensorStart(builder)
            tflite.TensorAddShape(builder, shape_vec)
            tflite.TensorAddType(builder, t.Type())
            tflite.TensorAddBuffer(builder, t.Buffer())
            tflite.TensorAddName(builder, name)
            if quant_offset:
                tflite.TensorAddQuantization(builder, quant_offset)
            if sparsity_offset:
                tflite.TensorAddSparsity(builder, sparsity_offset)
            tflite.TensorAddIsVariable(builder, t.IsVariable())
            tflite.TensorAddHasRank(builder, t.HasRank())
            if shape_sig_vec:
                tflite.TensorAddShapeSignature(builder, shape_sig_vec)

            tensor_offsets.append(tflite.TensorEnd(builder))

        # 2. Append newly registered tensors
        orig_buffers_count = orig_model.BuffersLength()
        for name_str, info in self._new_tensors.items():
            if name_str in tensor_name_to_idx:
                continue

            new_idx = len(tensor_offsets)
            tensor_name_to_idx[name_str] = new_idx

            name = builder.CreateString(name_str)
            shape = info["shape"]
            dtype_val = info["dtype"]
            dtype_code = _NP_DTYPE_TO_TFLITE.get(dtype_val, 0)

            buffer_data = info["buffer_data"]
            if buffer_data is not None:
                self._new_buffers.append(buffer_data)
                buf_idx = orig_buffers_count + len(self._new_buffers) - 1
            else:
                buf_idx = 0

            tflite.TensorStartShapeVector(builder, len(shape))
            for s in reversed(shape):
                builder.PrependInt32(s)
            shape_vec = builder.EndVector()

            tflite.TensorStart(builder)
            tflite.TensorAddShape(builder, shape_vec)
            tflite.TensorAddType(builder, dtype_code)
            tflite.TensorAddBuffer(builder, buf_idx)
            tflite.TensorAddName(builder, name)
            tflite.TensorAddIsVariable(builder, False)
            tflite.TensorAddHasRank(builder, True)
            tensor_offsets.append(tflite.TensorEnd(builder))

        tflite.SubGraphStartTensorsVector(builder, len(tensor_offsets))
        for t in reversed(tensor_offsets):
            builder.PrependUOffsetTRelative(t)
        return builder.EndVector(), tensor_name_to_idx

    def _build_operators(
        self,
        builder: flatbuffers.Builder,
        orig_subgraph: Any,
        opcode_map: dict[str, int],
        tensor_name_to_idx: dict[str, int],
    ) -> int:
        all_ops = []

        # 1. Gather original kept operators
        for i in range(orig_subgraph.OperatorsLength()):
            op = orig_subgraph.Operators(i)
            node_name = self._parser.nodes[i].name

            if node_name in self._to_remove:
                continue

            inputs = [op.Inputs(j) for j in range(op.InputsLength())]
            outputs = [op.Outputs(j) for j in range(op.OutputsLength())]

            all_ops.append(
                {
                    "is_original": True,
                    "op": op,
                    "node_name": node_name,
                    "inputs": inputs,
                    "outputs": outputs,
                    "opcode_idx": op.OpcodeIndex(),
                    "opt_type": op.BuiltinOptionsType(),
                    "opt_offset": op.BuiltinOptions(),
                }
            )

        # 2. Gather new operators
        for item in self._to_insert:
            op_type = item["op_type"]
            inputs_names = item["inputs"]
            outputs_names = item["outputs"]
            opcode_idx = opcode_map[op_type]

            inputs_idx = [
                tensor_name_to_idx[n] for n in inputs_names if n in tensor_name_to_idx
            ]
            outputs_idx = [
                tensor_name_to_idx[n] for n in outputs_names if n in tensor_name_to_idx
            ]

            attrs = item.get("attrs", {})
            opt_type_code, opt_serializer = _OP_TYPE_TO_BUILTIN_OPTIONS.get(
                op_type, (0, None)
            )
            opt_offset = 0
            if opt_serializer is not None:
                opt_offset = opt_serializer(builder, attrs)

            opt_type_code = attrs.get("builtin_options_type", opt_type_code)
            opt_offset = attrs.get("builtin_options", opt_offset)

            all_ops.append(
                {
                    "is_original": False,
                    "node_name": item["name"],
                    "inputs": inputs_idx,
                    "outputs": outputs_idx,
                    "opcode_idx": opcode_idx,
                    "opt_type": opt_type_code,
                    "opt_offset": opt_offset,
                }
            )

        # 3. Topological Sort using NetworkX
        g = nx.DiGraph()
        for idx in range(len(all_ops)):
            g.add_node(idx)

        for idx_a, op_a in enumerate(all_ops):
            for out_tensor in op_a["outputs"]:
                if out_tensor == -1:
                    continue
                for idx_b, op_b in enumerate(all_ops):
                    if idx_a == idx_b:
                        continue
                    if out_tensor in op_b["inputs"]:
                        g.add_edge(idx_a, idx_b)

        try:
            sorted_indices = list(nx.topological_sort(g))
            sorted_ops = [all_ops[idx] for idx in sorted_indices]
            log.info("Topological sort completed successfully.")
        except nx.NetworkXCyclicError:
            log.warning(
                "Cycle detected during rewrite! Falling back to unsorted order."
            )
            sorted_ops = all_ops

        # 4. Serialize sorted operators
        op_offsets = []
        for op in sorted_ops:
            inputs = op["inputs"]
            outputs = op["outputs"]

            tflite.OperatorStartInputsVector(builder, len(inputs))
            for x in reversed(inputs):
                builder.PrependInt32(x)
            inputs_vec = builder.EndVector()

            tflite.OperatorStartOutputsVector(builder, len(outputs))
            for x in reversed(outputs):
                builder.PrependInt32(x)
            outputs_vec = builder.EndVector()

            tflite.OperatorStart(builder)
            tflite.OperatorAddOpcodeIndex(builder, op["opcode_idx"])
            tflite.OperatorAddInputs(builder, inputs_vec)
            tflite.OperatorAddOutputs(builder, outputs_vec)
            if op["opt_type"]:
                tflite.OperatorAddBuiltinOptionsType(builder, op["opt_type"])
            if op["opt_offset"]:
                tflite.OperatorAddBuiltinOptions(builder, op["opt_offset"])
            op_offsets.append(tflite.OperatorEnd(builder))

        tflite.SubGraphStartOperatorsVector(builder, len(op_offsets))
        for o in reversed(op_offsets):
            builder.PrependUOffsetTRelative(o)
        return builder.EndVector()

    def _serialize_opcodes(
        self, builder: flatbuffers.Builder, opcode_map: dict[str, int]
    ) -> int:
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
        return builder.EndVector()

    def _build_buffers(self, builder: flatbuffers.Builder, orig_model: Any) -> int:
        buffer_offsets = []
        for i in range(orig_model.BuffersLength()):
            b = orig_model.Buffers(i)
            data = b.DataAsNumpy()
            if data is not None and hasattr(data, "__len__") and len(data) > 0:
                data_vec = builder.CreateByteVector(bytes(data))
            else:
                data_vec = 0

            tflite.BufferStart(builder)
            if data_vec:
                tflite.BufferAddData(builder, data_vec)
            buffer_offsets.append(tflite.BufferEnd(builder))

        # Append new buffers from self._new_buffers
        for data in self._new_buffers:
            data_vec = builder.CreateByteVector(bytes(data))
            tflite.BufferStart(builder)
            tflite.BufferAddData(builder, data_vec)
            buffer_offsets.append(tflite.BufferEnd(builder))

        tflite.ModelStartBuffersVector(builder, len(buffer_offsets))
        for b in reversed(buffer_offsets):
            builder.PrependUOffsetTRelative(b)
        return builder.EndVector()

    def _copy_subgraph(self, builder: flatbuffers.Builder, sub: Any) -> int:
        """Copy a secondary subgraph as-is."""
        # Tensors
        tensor_offsets = []
        for i in range(sub.TensorsLength()):
            t = sub.Tensors(i)
            name_str = t.Name().decode("utf-8") if t.Name() else f"tensor_{i}"
            name = builder.CreateString(name_str)
            shape = [t.Shape(j) for j in range(t.ShapeLength())]

            tflite.TensorStartShapeVector(builder, len(shape))
            for s in reversed(shape):
                builder.PrependInt32(s)
            shape_vec = builder.EndVector()

            quant_offset = _copy_quantization(builder, t.Quantization())
            sparsity_offset = _copy_sparsity(builder, t.Sparsity())

            tflite.TensorStart(builder)
            tflite.TensorAddShape(builder, shape_vec)
            tflite.TensorAddType(builder, t.Type())
            tflite.TensorAddBuffer(builder, t.Buffer())
            tflite.TensorAddName(builder, name)
            if quant_offset:
                tflite.TensorAddQuantization(builder, quant_offset)
            if sparsity_offset:
                tflite.TensorAddSparsity(builder, sparsity_offset)
            tflite.TensorAddIsVariable(builder, t.IsVariable())
            tflite.TensorAddHasRank(builder, t.HasRank())
            tensor_offsets.append(tflite.TensorEnd(builder))

        tflite.SubGraphStartTensorsVector(builder, len(tensor_offsets))
        for t in reversed(tensor_offsets):
            builder.PrependUOffsetTRelative(t)
        tensors_vec = builder.EndVector()

        # Operators
        op_offsets = []
        for i in range(sub.OperatorsLength()):
            op = sub.Operators(i)
            inputs = [op.Inputs(j) for j in range(op.InputsLength())]
            outputs = [op.Outputs(j) for j in range(op.OutputsLength())]

            tflite.OperatorStartInputsVector(builder, len(inputs))
            for x in reversed(inputs):
                builder.PrependInt32(x)
            inputs_vec = builder.EndVector()

            tflite.OperatorStartOutputsVector(builder, len(outputs))
            for x in reversed(outputs):
                builder.PrependInt32(x)
            outputs_vec = builder.EndVector()

            tflite.OperatorStart(builder)
            tflite.OperatorAddOpcodeIndex(builder, op.OpcodeIndex())
            tflite.OperatorAddInputs(builder, inputs_vec)
            tflite.OperatorAddOutputs(builder, outputs_vec)
            tflite.OperatorAddBuiltinOptionsType(builder, op.BuiltinOptionsType())
            tflite.OperatorAddBuiltinOptions(builder, op.BuiltinOptions())
            op_offsets.append(tflite.OperatorEnd(builder))

        tflite.SubGraphStartOperatorsVector(builder, len(op_offsets))
        for o in reversed(op_offsets):
            builder.PrependUOffsetTRelative(o)
        ops_vec = builder.EndVector()

        # Inputs/Outputs
        sub_inputs = [sub.Inputs(i) for i in range(sub.InputsLength())]
        tflite.SubGraphStartInputsVector(builder, len(sub_inputs))
        for i in reversed(sub_inputs):
            builder.PrependInt32(i)
        sub_inputs_vec = builder.EndVector()

        sub_outputs = [sub.Outputs(i) for i in range(sub.OutputsLength())]
        tflite.SubGraphStartOutputsVector(builder, len(sub_outputs))
        for o in reversed(sub_outputs):
            builder.PrependInt32(o)
        sub_outputs_vec = builder.EndVector()

        name_offset = builder.CreateString(
            sub.Name().decode("utf-8") if sub.Name() else "subgraph"
        )

        tflite.SubGraphStart(builder)
        tflite.SubGraphAddTensors(builder, tensors_vec)
        tflite.SubGraphAddInputs(builder, sub_inputs_vec)
        tflite.SubGraphAddOutputs(builder, sub_outputs_vec)
        tflite.SubGraphAddOperators(builder, ops_vec)
        tflite.SubGraphAddName(builder, name_offset)
        return tflite.SubGraphEnd(builder)

    def build(self, output_path: str | None = None) -> bytes:  # noqa: PLR0915
        """Apply staged edits and return modified model flatbuffer bytes."""
        if not TFLITE_AVAILABLE:
            msg = "flatbuffers or tflite package not found. Cannot perform build."
            raise RuntimeError(msg)

        if not self._to_remove and not self._to_insert:
            msg = "No edits staged. Call replace(), delete(), or insert_before() first."
            raise ValueError(msg)

        # Load original model
        orig_model: Any
        if self._parser.path:
            buf = Path(self._parser.path).read_bytes()
            orig_model = tflite.Model.GetRootAsModel(buf, 0)
        elif isinstance(self._parser.source, bytes):
            orig_model = tflite.Model.GetRootAsModel(self._parser.source, 0)
        else:
            orig_model = cast(Any, self._parser.source)

        orig_subgraph = orig_model.Subgraphs(0)
        builder = flatbuffers.Builder(1024 * 1024)

        # 1. Operator Codes
        opcode_map = self._prepare_opcodes(orig_model)

        # 2. Tensors Vector
        tensors_vec, tensor_name_to_idx = self._build_tensors(
            builder, orig_subgraph, orig_model
        )

        # 3. Operators Vector
        ops_vec = self._build_operators(
            builder, orig_subgraph, opcode_map, tensor_name_to_idx
        )

        # 4. Inputs/Outputs vectors for SubGraph
        sub_inputs = [
            orig_subgraph.Inputs(i) for i in range(orig_subgraph.InputsLength())
        ]
        tflite.SubGraphStartInputsVector(builder, len(sub_inputs))
        for i in reversed(sub_inputs):
            builder.PrependInt32(i)
        sub_inputs_vec = builder.EndVector()

        sub_outputs = [
            orig_subgraph.Outputs(i) for i in range(orig_subgraph.OutputsLength())
        ]
        tflite.SubGraphStartOutputsVector(builder, len(sub_outputs))
        for o in reversed(sub_outputs):
            builder.PrependInt32(o)
        sub_outputs_vec = builder.EndVector()

        # 5. Build SubGraph
        subgraphs_offsets = []

        subgraph_name = builder.CreateString(
            orig_subgraph.Name().decode("utf-8")
            if orig_subgraph.Name()
            else "rewritten_subgraph"
        )
        tflite.SubGraphStart(builder)
        tflite.SubGraphAddTensors(builder, tensors_vec)
        tflite.SubGraphAddInputs(builder, sub_inputs_vec)
        tflite.SubGraphAddOutputs(builder, sub_outputs_vec)
        tflite.SubGraphAddOperators(builder, ops_vec)
        tflite.SubGraphAddName(builder, subgraph_name)
        subgraphs_offsets.append(tflite.SubGraphEnd(builder))

        # Copy secondary subgraphs if they exist
        for i in range(1, orig_model.SubgraphsLength()):
            sec_sub = orig_model.Subgraphs(i)
            subgraphs_offsets.append(self._copy_subgraph(builder, sec_sub))

        tflite.ModelStartSubgraphsVector(builder, len(subgraphs_offsets))
        for off in reversed(subgraphs_offsets):
            builder.PrependUOffsetTRelative(off)
        subgraphs_vec = builder.EndVector()

        # 6. Rebuild OperatorCodes (including new ones)
        opcode_vec = self._serialize_opcodes(builder, opcode_map)

        # 7. Rebuild Buffers
        buffers_vec = self._build_buffers(builder, orig_model)

        # 8. Rebuild SignatureDefs
        sig_offsets = []
        for i in range(orig_model.SignatureDefsLength()):
            sig = orig_model.SignatureDefs(i)
            sig_offsets.append(_copy_signature_def(builder, sig, tensor_name_to_idx))

        sigs_vec = 0
        if sig_offsets:
            tflite.ModelStartSignatureDefsVector(builder, len(sig_offsets))
            for off in reversed(sig_offsets):
                builder.PrependUOffsetTRelative(off)
            sigs_vec = builder.EndVector()

        # 9. Rebuild Metadata
        meta_offsets = []
        for i in range(orig_model.MetadataLength()):
            meta = orig_model.Metadata(i)
            meta_offsets.append(_copy_metadata(builder, meta))

        meta_vec = 0
        if meta_offsets:
            tflite.ModelStartMetadataVector(builder, len(meta_offsets))
            for off in reversed(meta_offsets):
                builder.PrependUOffsetTRelative(off)
            meta_vec = builder.EndVector()

        # 10. Copy Metadata Buffer indices
        meta_buf_indices = (
            [
                orig_model.MetadataBuffer(i)
                for i in range(orig_model.MetadataBufferLength())
            ]
            if not orig_model.MetadataBufferIsNone()
            else []
        )
        meta_buf_vec = 0
        if meta_buf_indices:
            tflite.ModelStartMetadataBufferVector(builder, len(meta_buf_indices))
            for val in reversed(meta_buf_indices):
                builder.PrependInt32(val)
            meta_buf_vec = builder.EndVector()

        # 11. Build Model
        desc = builder.CreateString("rewritten by neuron_toolkit")
        tflite.ModelStart(builder)
        tflite.ModelAddVersion(builder, orig_model.Version())
        tflite.ModelAddOperatorCodes(builder, opcode_vec)
        tflite.ModelAddSubgraphs(builder, subgraphs_vec)
        tflite.ModelAddBuffers(builder, buffers_vec)
        tflite.ModelAddDescription(builder, desc)
        if sigs_vec:
            tflite.ModelAddSignatureDefs(builder, sigs_vec)
        if meta_vec:
            tflite.ModelAddMetadata(builder, meta_vec)
        if meta_buf_vec:
            tflite.ModelAddMetadataBuffer(builder, meta_buf_vec)
        model_offset = tflite.ModelEnd(builder)

        builder.Finish(model_offset)
        output_data = bytes(builder.Output())

        result_path = output_path
        if not result_path and self._parser.path:
            result_path = self._parser.path + ".rewritten.tflite"

        if result_path:
            Path(result_path).write_bytes(output_data)
            log.info("build(): saved rewritten model to %r", result_path)

        self.reset()
        return output_data
