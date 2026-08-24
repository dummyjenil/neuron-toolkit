"""ONNX specific graph rewriter."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import onnx
from onnx import helper
from onnx.onnx_pb import NodeProto

from neuron_toolkit import MatchResult

if TYPE_CHECKING:
    from neuron_toolkit.backends.onnx.parser import ONNXParser

from neuron_toolkit.backends.base import BaseRewriter

log = logging.getLogger("neuron_toolkit.backends.onnx.rewriter")


class ONNXRewriter(BaseRewriter):
    """Staged graph rewriter for ONNX models."""

    def __init__(self, parser: ONNXParser) -> None:
        """Initialize with an ONNXParser instance."""
        self._parser = parser
        self._to_remove_ids: set[int] = set()  # node object IDs to drop
        self._to_insert: list[NodeProto] = []  # new nodes to add
        self._new_initializers: dict[str, onnx.TensorProto] = {}  # new constants

    # ------------------------------------------------------------------
    # Edit API
    # ------------------------------------------------------------------

    def replace(
        self,
        nodes: Sequence[object],
        new_op: str,
        inputs: list[str],
        outputs: list[str],
        name: str | None = None,
        **attrs: object,
    ) -> ONNXRewriter:
        """Replace *nodes* with a single new node of type *new_op*."""
        for n in nodes:
            self._to_remove_ids.add(id(n))
        node_name = name or f"{new_op}_{len(self._to_insert)}_rewrite"
        new_node = helper.make_node(
            new_op,
            inputs=inputs,
            outputs=outputs,
            name=node_name,
            **cast(dict[str, Any], attrs),
        )
        self._to_insert.append(new_node)
        log.debug(
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
    ) -> ONNXRewriter:
        """Convenience wrapper: replace all nodes in *result* with a new op.

        *inputs* defaults to the inputs of the boundary nodes in the result.
        *outputs* defaults to the outputs of the result's root node.
        """
        # Improved boundary detection:
        # Inputs are tensors consumed by result nodes produced outside result.
        internal_outputs = {
            out for n in result.nodes for out in getattr(n, "output", [])
        }
        if inputs is None:
            ins = []
            for n in result.nodes:
                for inp in getattr(n, "input", []):
                    if inp and inp not in internal_outputs and inp not in ins:
                        # Only include if initializer/graph input
                        ins.append(inp)
            inputs = ins

        if outputs is None:
            # Outputs are tensors produced by result nodes consumed outside,
            # or they are graph outputs.
            outputs = list(getattr(result.start, "output", []))

        return self.replace(result.nodes, new_op, inputs, outputs, name=name, **attrs)

    def delete(self, nodes: Sequence[object]) -> ONNXRewriter:
        """Remove *nodes* from the graph entirely."""
        for n in nodes:
            self._to_remove_ids.add(id(n))
        log.debug("delete(): scheduled removal of %d node(s)", len(nodes))
        return self

    def insert_before(
        self,
        target_node: object,
        new_op: str,
        inputs: list[str],
        outputs: list[str],
        name: str | None = None,
        **attrs: object,
    ) -> ONNXRewriter:
        """Insert a new node whose outputs feed *target_node*."""
        node_name = name or f"{new_op}_{len(self._to_insert)}_insert"
        new_node = helper.make_node(
            new_op,
            inputs=inputs,
            outputs=outputs,
            name=node_name,
            **cast(dict[str, Any], attrs),
        )
        self._to_insert.append(new_node)
        target_name = getattr(target_node, "name", "<unnamed>")
        log.debug(
            "insert_before(%r): scheduled insertion of %r",
            target_name,
            node_name,
        )
        return self

    def reset(self) -> ONNXRewriter:
        """Discard all pending edits."""
        self._to_remove_ids.clear()
        self._to_insert.clear()
        self._new_initializers.clear()
        return self

    def register_initializer(self, name: str, value: Any) -> ONNXRewriter:
        """Register a new constant tensor that will be added to the rewritten graph.

        Useful for passes that need to inject folded weights/bias tensors.
        """
        from onnx import numpy_helper

        self._new_initializers[name] = numpy_helper.from_array(value, name=name)
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, output_path: str | None = None) -> onnx.ModelProto:
        """Apply all staged edits and return the new ModelProto."""
        if not self._to_remove_ids and not self._to_insert:
            msg = "No edits staged. Call replace(), delete(), or insert_before() first."
            raise ValueError(msg)

        orig_graph = self._parser.model.graph

        # Keep nodes not scheduled for removal
        kept_nodes = [n for n in orig_graph.node if id(n) not in self._to_remove_ids]
        # Append new nodes
        all_nodes = kept_nodes + self._to_insert

        # Fast native Kahn's algorithm topological sort
        producer_map: dict[str, int] = {
            out: j for j, p in enumerate(all_nodes) for out in p.output if out
        }

        n_count = len(all_nodes)
        in_degree = [0] * n_count
        adj: list[list[int]] = [[] for _ in range(n_count)]

        for i, n in enumerate(all_nodes):
            for inp in n.input:
                if not inp:
                    continue
                p_idx = producer_map.get(inp, -1)
                if p_idx not in (-1, i):
                    adj[p_idx].append(i)
                    in_degree[i] += 1

        zero_in_degree = [i for i in range(n_count) if in_degree[i] == 0]
        sorted_indices: list[int] = []

        while zero_in_degree:
            curr = zero_in_degree.pop()
            sorted_indices.append(curr)
            for nxt in adj[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    zero_in_degree.append(nxt)

        if len(sorted_indices) == n_count:
            final_nodes = [cast(NodeProto, all_nodes[idx]) for idx in sorted_indices]
        else:
            log.warning("Cycle detected during rewrite! Falling back to unsorted nodes.")
            final_nodes = all_nodes

        new_graph = helper.make_graph(
            nodes=final_nodes,
            name=orig_graph.name or "rewritten",
            inputs=list(orig_graph.input),
            outputs=list(orig_graph.output),
            initializer=list(orig_graph.initializer)
            + list(self._new_initializers.values()),
        )

        new_model = helper.make_model(new_graph)
        new_model.ir_version = self._parser.model.ir_version
        del new_model.opset_import[:]
        new_model.opset_import.extend(self._parser.model.opset_import)

        # Copy other metadata
        for attr in [
            "producer_name",
            "producer_version",
            "domain",
            "model_version",
            "doc_string",
        ]:
            if hasattr(self._parser.model, attr):
                setattr(new_model, attr, getattr(self._parser.model, attr))

        try:
            new_model = onnx.shape_inference.infer_shapes(new_model)
        except (
            ValueError,
            TypeError,
            RuntimeError,
            onnx.shape_inference.InferenceError,
        ) as exc:
            log.warning("Shape inference failed after rewrite: %s", exc)

        if output_path:
            onnx.save(new_model, output_path)
            log.info("Rewritten model saved to %r", output_path)

        log.debug(
            "build(): removed %d node(s), inserted %d node(s), result has %d node(s)",
            len(self._to_remove_ids),
            len(self._to_insert),
            len(final_nodes),
        )

        # Clear staging after build
        self.reset()

        return new_model
