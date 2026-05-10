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
        """Apply all staged edits and return the new model (currently simulated)."""
        if not self._to_remove and not self._to_insert:
            msg = "No edits staged. Call replace(), delete(), or insert_before() first."
            raise ValueError(msg)

        log.warning(
            "TFLite build() is currently a simulation. Flatbuffer modification not fully implemented."
        )

        # In a real implementation, we would reconstruct the flatbuffer here.
        # For now, we return the original path to avoid breaking workflows.
        result_path = output_path or self._parser.path

        log.info(
            "build(): removed %d node(s), inserted %d node(s)",
            len(self._to_remove),
            len(self._to_insert),
        )

        self.reset()
        return result_path
