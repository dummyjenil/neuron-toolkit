"""TFLite backend rewriter."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from neuron_toolkit.backends.base import BaseRewriter

if TYPE_CHECKING:
    from neuron_toolkit.backends.tflite.parser import TFLiteNode, TFLiteParser
    from neuron_toolkit.pattern import MatchResult

log = logging.getLogger("neuron_toolkit.backends.tflite.rewriter")


class TFLiteRewriter(BaseRewriter):
    """Transformation engine for TFLite models.

    Currently, this rewriter primarily stages edits. Real flatbuffer modification
    is a complex task and is partially simulated.
    """

    def __init__(self, parser: TFLiteParser) -> None:
        """Initialize with a TFLiteParser instance."""
        self._parser = parser
        self._to_remove: set[str] = set()  # node names to drop
        self._to_insert: list[dict[str, Any]] = []  # new nodes to add

    def replace(
        self,
        nodes: Sequence[TFLiteNode],
        new_op: str,
        inputs: list[str],
        outputs: list[str],
        name: str | None = None,
        **attrs: Any,
    ) -> TFLiteRewriter:
        """Replace *nodes* with a single new node of type *new_op*."""
        for n in nodes:
            self._to_remove.add(n.name)

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
        **attrs: Any,
    ) -> TFLiteRewriter:
        """Convenience wrapper: replace all nodes in *result* with a new op.

        *inputs* defaults to the inputs of the boundary nodes in the result.
        *outputs* defaults to the outputs of the result's root node.
        """
        internal_outputs = {out for n in result.nodes for out in n.output}
        if inputs is None:
            ins = []
            for n in result.nodes:
                for inp in n.input:
                    if inp and inp not in internal_outputs and inp not in ins:
                        ins.append(inp)
            inputs = ins

        if outputs is None:
            outputs = list(result.start.output)

        return self.replace(result.nodes, new_op, inputs, outputs, name=name, **attrs)

    def delete(self, nodes: Sequence[TFLiteNode]) -> TFLiteRewriter:
        """Remove *nodes* from the graph entirely."""
        for n in nodes:
            self._to_remove.add(n.name)
        log.info("delete(): scheduled removal of %d node(s)", len(nodes))
        return self

    def insert_before(
        self,
        target_node: TFLiteNode,
        new_op: str,
        inputs: list[str],
        outputs: list[str],
        name: str | None = None,
        **attrs: Any,
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
        log.info(
            "insert_before(%r): scheduled insertion of %r", target_node.name, node_name
        )
        return self

    def reset(self) -> TFLiteRewriter:
        """Discard all pending edits."""
        self._to_remove.clear()
        self._to_insert.clear()
        return self

    def build(self, output_path: str | None = None) -> str:
        """Apply all staged edits and return the new model file path."""
        if not self._to_remove and not self._to_insert:
            msg = "No edits staged. Call replace(), delete(), or insert_before() first."
            raise ValueError(msg)

        try:
            import flatbuffers
            import tflite
        except ImportError:
            log.error(
                "flatbuffers or tflite package not found. Cannot perform real build."
            )
            return self._parser.path

        # Load original model
        with open(self._parser.path, "rb") as f:
            buf = f.read()
        orig_model = tflite.Model.GetRootAsModel(buf, 0)
        orig_subgraph = orig_model.Subgraphs(0)

        builder = flatbuffers.Builder(1024 * 1024)

        # 1. Prepare Operator Codes
        opcode_map: dict[str, int] = {}
        for i in range(orig_model.OperatorCodesLength()):
            oc = orig_model.OperatorCodes(i)
            builtin = oc.BuiltinCode()

            # Map builtin code to its string name
            op_name = "UNKNOWN"
            if builtin != tflite.BuiltinOperator.CUSTOM:
                for k, v in tflite.BuiltinOperator.__dict__.items():
                    if isinstance(v, int) and v == builtin and not k.startswith("__"):
                        op_name = k
                        break
            else:
                op_name = oc.CustomCode().decode("utf-8")

            opcode_map[op_name] = i
        # Handle new opcodes from staged insertions
        for item in self._to_insert:
            op_type = item["op_type"]
            if op_type not in opcode_map:
                new_idx = len(opcode_map)
                opcode_map[op_type] = new_idx

        # 2. Build Tensors Vector
        tensor_offsets = []
        tensor_name_to_idx = {}
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

            tflite.TensorStart(builder)
            tflite.TensorAddShape(builder, shape_vec)
            tflite.TensorAddType(builder, t.Type())
            tflite.TensorAddBuffer(builder, t.Buffer())
            tflite.TensorAddName(builder, name)
            tensor_offsets.append(tflite.TensorEnd(builder))

        tflite.SubGraphStartTensorsVector(builder, len(tensor_offsets))
        for t in reversed(tensor_offsets):
            builder.PrependUOffsetTRelative(t)
        tensors_vec = builder.EndVector()

        # 3. Build Operators Vector
        op_offsets = []

        # Add original operators not marked for removal
        for i in range(orig_subgraph.OperatorsLength()):
            op = orig_subgraph.Operators(i)
            node_name = self._parser.nodes[i].name

            if node_name in self._to_remove:
                continue

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
            # We skip BuiltinOptions for now to keep it simple, or reuse them
            tflite.OperatorAddBuiltinOptionsType(builder, op.BuiltinOptionsType())
            tflite.OperatorAddBuiltinOptions(builder, op.BuiltinOptions())
            op_offsets.append(tflite.OperatorEnd(builder))

        # Add new operators from staged edits
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

            tflite.OperatorStartInputsVector(builder, len(inputs_idx))
            for x in reversed(inputs_idx):
                builder.PrependInt32(x)
            inputs_vec = builder.EndVector()

            tflite.OperatorStartOutputsVector(builder, len(outputs_idx))
            for x in reversed(outputs_idx):
                builder.PrependInt32(x)
            outputs_vec = builder.EndVector()

            tflite.OperatorStart(builder)
            tflite.OperatorAddOpcodeIndex(builder, opcode_idx)
            tflite.OperatorAddInputs(builder, inputs_vec)
            tflite.OperatorAddOutputs(builder, outputs_vec)
            op_offsets.append(tflite.OperatorEnd(builder))

        tflite.SubGraphStartOperatorsVector(builder, len(op_offsets))
        for o in reversed(op_offsets):
            builder.PrependUOffsetTRelative(o)
        ops_vec = builder.EndVector()

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
        subgraph_name = builder.CreateString("rewritten_subgraph")
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

        # 6. Rebuild OperatorCodes (including new ones)
        opcode_offsets = []
        # Sort opcode_map by index to maintain order
        sorted_opcodes = sorted(opcode_map.items(), key=lambda x: x[1])
        for name, idx in sorted_opcodes:
            # Explicitly check if it's a known builtin
            if hasattr(tflite.BuiltinOperator, name) and not name.startswith("__"):
                builtin_code = getattr(tflite.BuiltinOperator, name)
            else:
                # Default to CUSTOM (usually 255, but use the constant)
                builtin_code = tflite.BuiltinOperator.CUSTOM

            custom_code_offset = 0
            if builtin_code == tflite.BuiltinOperator.CUSTOM:
                custom_code_offset = builder.CreateString(name)

            tflite.OperatorCodeStart(builder)

            # For compatibility, many models set both
            if builtin_code < 127:
                tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, builtin_code)

            tflite.OperatorCodeAddBuiltinCode(builder, builtin_code)

            if custom_code_offset:
                tflite.OperatorCodeAddCustomCode(builder, custom_code_offset)
            opcode_offsets.append(tflite.OperatorCodeEnd(builder))

        tflite.ModelStartOperatorCodesVector(builder, len(opcode_offsets))
        for o in reversed(opcode_offsets):
            builder.PrependUOffsetTRelative(o)
        opcode_vec = builder.EndVector()

        # 7. Rebuild Buffers
        buffer_offsets = []
        for i in range(orig_model.BuffersLength()):
            b = orig_model.Buffers(i)
            data = b.DataAsNumpy()
            # Use hasattr to safely check for length
            if data is not None and hasattr(data, "__len__") and len(data) > 0:
                data_vec = builder.CreateByteVector(bytes(data))
            else:
                data_vec = 0

            tflite.BufferStart(builder)
            if data_vec:
                tflite.BufferAddData(builder, data_vec)
            buffer_offsets.append(tflite.BufferEnd(builder))

        tflite.ModelStartBuffersVector(builder, len(buffer_offsets))
        for b in reversed(buffer_offsets):
            builder.PrependUOffsetTRelative(b)
        buffers_vec = builder.EndVector()

        # 8. Build Model
        desc = builder.CreateString("rewritten by neuron_toolkit")
        tflite.ModelStart(builder)
        tflite.ModelAddVersion(builder, orig_model.Version())
        tflite.ModelAddOperatorCodes(builder, opcode_vec)
        tflite.ModelAddSubgraphs(builder, subgraphs_vec)
        tflite.ModelAddBuffers(builder, buffers_vec)
        tflite.ModelAddDescription(builder, desc)
        model_offset = tflite.ModelEnd(builder)

        builder.Finish(model_offset)

        result_path = output_path or (self._parser.path + ".rewritten.tflite")
        with open(result_path, "wb") as f:
            f.write(builder.Output())

        log.info("build(): saved rewritten model to %r", result_path)
        self.reset()
        return result_path
