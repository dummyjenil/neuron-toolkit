"""TFLite backend parser."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from neuron_toolkit.backends.base import BaseParser, BaseRewriter
from neuron_toolkit.query import NeuronQuery

if TYPE_CHECKING:
    from neuron_toolkit.pattern import MatchResult, Pattern

log = logging.getLogger("neuron_toolkit.backends.tflite")


class LazyTensorMap(dict):
    """Lazy-loading map for TFLite tensors."""

    def __init__(self, model: Any, subgraph: Any) -> None:
        super().__init__()
        self._model = model
        self._subgraph = subgraph
        self._cache: dict[str, Any] = {}
        self._name_to_idx: dict[str, int] = {}

        for i in range(subgraph.TensorsLength()):
            name = subgraph.Tensors(i).Name().decode("utf-8")
            self._name_to_idx[name] = i

    def __getitem__(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]
        if key in self._name_to_idx:
            idx = self._name_to_idx[key]
            tensor = self._subgraph.Tensors(idx)
            buffer_idx = tensor.Buffer()
            if buffer_idx > 0:
                buffer = self._model.Buffers(buffer_idx)
                data = buffer.DataAsNumpy()
                if data is not None:
                    self._cache[key] = data
                    return data
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        return (
            key in self._name_to_idx or key in self._cache or super().__contains__(key)
        )

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __len__(self) -> int:
        return len(self._name_to_idx)


@dataclass
class TFLiteNode:
    """Mock-up of a TFLite node that looks like an ONNX node for the query engine."""

    op_type: str
    name: str
    input: list[str]
    output: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)


class TFLiteParser(BaseParser):
    """Load a TFLite model and expose it for querying and pattern matching."""

    def __init__(self, tflite_path: str, **kwargs: Any) -> None:
        """Initialize the TFLiteParser."""
        log.info("Loading TFLite model from %r", tflite_path)
        self.path = tflite_path

        self.nodes: list[TFLiteNode] = []
        self.tensor_map: LazyTensorMap | dict[str, Any] = {}
        self.graph_inputs: set[str] = set()
        self.graph_outputs: set[str] = set()
        self.shape_info: dict[str, Any] = {}

        self._load_model()

    def _load_model(self) -> None:
        """Parse TFLite flatbuffer using the 'tflite' package."""
        try:
            import tflite
        except ImportError:
            log.error("TFLite package not found. Please install 'tflite'.")
            return

        from neuron_toolkit.backends.tflite.utils import (
            _build_shape_info,
            _get_tflite_attr,
        )

        with open(self.path, "rb") as f:
            buf = f.read()
            model = tflite.Model.GetRootAsModel(buf, 0)

        # TFLite can have multiple subgraphs, we take the primary one (index 0)
        subgraph = model.Subgraphs(0)

        # Build lazy tensor map
        self.tensor_map = LazyTensorMap(model, subgraph)

        # Build shape info
        self.shape_info = _build_shape_info(model, subgraph)

        # Build nodes
        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)
            opcode_idx = op.OpcodeIndex()
            opcode = model.OperatorCodes(opcode_idx)

            # Get OpType
            builtin_code = opcode.BuiltinCode()

            if builtin_code != tflite.BuiltinOperator.CUSTOM:
                op_type = [
                    k
                    for k, v in tflite.BuiltinOperator.__dict__.items()
                    if v == builtin_code and not k.startswith("__")
                ][0]
            else:
                op_type = opcode.CustomCode().decode("utf-8")

            # Inputs and Outputs
            inputs = []
            for j in range(op.InputsLength()):
                t_idx = op.Inputs(j)
                if t_idx != -1:
                    inputs.append(subgraph.Tensors(t_idx).Name().decode("utf-8"))

            outputs = []
            for j in range(op.OutputsLength()):
                t_idx = op.Outputs(j)
                if t_idx != -1:
                    outputs.append(subgraph.Tensors(t_idx).Name().decode("utf-8"))

            # Attributes
            attrs = _get_tflite_attr(op, op_type)

            self.nodes.append(
                TFLiteNode(
                    op_type=op_type,
                    name=f"{op_type}_{i}",
                    input=inputs,
                    output=outputs,
                    attrs=attrs,
                )
            )

        # Graph inputs/outputs
        for i in range(subgraph.InputsLength()):
            self.graph_inputs.add(
                subgraph.Tensors(subgraph.Inputs(i)).Name().decode("utf-8")
            )
        for i in range(subgraph.OutputsLength()):
            self.graph_outputs.add(
                subgraph.Tensors(subgraph.Outputs(i)).Name().decode("utf-8")
            )

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
        start_node: str | TFLiteNode | None = None,
        end_node: str | TFLiteNode | None = None,
    ) -> MatchResult | None:
        """Create a PatternDetector bound to this model and call match()."""
        from neuron_toolkit._utils import _GraphShim
        from neuron_toolkit.pattern import PatternDetector

        shim = _GraphShim(self.nodes, self.tensor_map, self.shape_info, backend=self)
        det = PatternDetector(shim, start_node=start_node, end_node=end_node)
        return det.match(pattern)

    def get_node_attrs(self, node: TFLiteNode) -> dict[str, Any]:
        """Extract attributes from a TFLite node."""
        return node.attrs

    def is_constant_node(self, node: TFLiteNode) -> bool:
        """TFLite doesn't typically use explicit Constant nodes in the same way."""
        # Some TFLite exporters might use a CONST op
        return node.op_type == "CONST"

    def get_constant_value(self, node: TFLiteNode) -> Any:
        """Extract value from TFLite CONST node if available."""
        return node.attrs.get("value")

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
        from neuron_toolkit.backends.tflite.rewriter import TFLiteRewriter

        return TFLiteRewriter(self)
