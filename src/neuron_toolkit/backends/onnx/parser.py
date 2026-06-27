"""ONNX backend parser."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import onnx
from onnx import ModelProto, numpy_helper
from onnx.onnx_pb import NodeProto

from neuron_toolkit._utils import ShapeInfo, _GraphShim
from neuron_toolkit.backends.base import BaseParser, BaseRewriter
from neuron_toolkit.backends.onnx.utils import _build_shape_info
from neuron_toolkit.query import NeuronQuery

if TYPE_CHECKING:
    import numpy as np

    from neuron_toolkit.pattern import MatchResult, Pattern

log = logging.getLogger("neuron_toolkit.backends.onnx")


class LazyTensorMap(dict[str, "np.ndarray"]):
    """Lazy-loading map for ONNX initializers."""

    def __init__(self, initializers: Sequence[onnx.TensorProto]) -> None:
        super().__init__()
        self._initializers = {t.name: t for t in initializers}
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._cache:
            return self._cache[key]
        if key in self._initializers:
            arr = numpy_helper.to_array(self._initializers[key])
            self._cache[key] = arr
            return arr
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        return (
            key in self._initializers or key in self._cache or super().__contains__(key)
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get tensor by key with optional default."""
        try:
            return self[key]
        except KeyError:
            return default

    def __len__(self) -> int:
        return len(self._initializers)


class ONNXParser(BaseParser):
    """Load an ONNX model and expose it for querying and pattern matching."""

    def __init__(
        self,
        onnx_source: str | bytes | ModelProto,
        *,
        infer_shapes: bool = True,
        **_kwargs: object,
    ) -> None:
        """Initialize the ONNXParser with model file, bytes, or ModelProto."""
        if isinstance(onnx_source, ModelProto):
            self.model = onnx_source
        elif isinstance(onnx_source, bytes):
            self.model = onnx.load_model_from_string(onnx_source)
        else:
            log.info("Loading ONNX model from %r", onnx_source)
            self.model = onnx.load(onnx_source)

        if infer_shapes:
            try:
                self.model = onnx.shape_inference.infer_shapes(self.model)
                log.debug("Shape inference completed.")
            except Exception as exc:  # noqa: BLE001
                log.warning("Shape inference failed: %s", exc)

        self.nodes: list[NodeProto] = list(self.model.graph.node)
        self.tensor_map = LazyTensorMap(self.model.graph.initializer)
        self.graph_inputs: set[str] = {i.name for i in self.model.graph.input}
        self.graph_outputs: set[str] = {o.name for o in self.model.graph.output}
        self.shape_info: ShapeInfo = _build_shape_info(self.model)

        log.info(
            "Model loaded: %d nodes, %d tensors, %d inputs, "
            "%d outputs, %d shape annotations",
            len(self.nodes),
            len(self.tensor_map),
            len(self.graph_inputs),
            len(self.graph_outputs),
            len(self.shape_info),
        )

    # ------------------------------------------------------------------

    def find(self) -> NeuronQuery:
        """Return an NeuronQuery over **all** nodes in the graph."""
        return NeuronQuery(
            list(self.nodes),
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
        from neuron_toolkit.pattern import PatternDetector  # noqa: PLC0415

        shim = _GraphShim(self.nodes, self.tensor_map, self.shape_info, backend=self)
        det = PatternDetector(shim, start_node=start_node, end_node=end_node)
        return det.match(pattern)

    def rewriter(self) -> BaseRewriter:
        """Return a rewriter bound to this model."""
        from neuron_toolkit.backends.onnx.rewriter import ONNXRewriter  # noqa: PLC0415

        return ONNXRewriter(self)

    def get_node_attrs(self, node: object) -> dict[str, object]:
        """Extract attributes from an ONNX node."""
        from neuron_toolkit.backends.onnx.utils import _node_attrs  # noqa: PLC0415

        return _node_attrs(cast(NodeProto, node))

    def is_constant_node(self, node: object) -> bool:
        """Check if node is an ONNX Constant."""
        return getattr(node, "op_type", None) == "Constant"

    def get_constant_value(self, node: object) -> object | None:
        """Extract value from ONNX Constant node."""
        from onnx import numpy_helper  # noqa: PLC0415

        for attr in getattr(node, "attribute", []):
            if attr.name == "value":
                return numpy_helper.to_array(attr.t)
        return None

    def summary(self) -> str:
        """Return a human-readable summary of the model."""
        op_counts = Counter(n.op_type for n in self.nodes)
        lines = [
            "ONNX model summary",
            f"  Nodes        : {len(self.nodes)}",
            f"  Tensors      : {len(self.tensor_map)}",
            f"  Graph inputs : {len(self.graph_inputs)}",
            f"  Graph outputs: {len(self.graph_outputs)}",
            f"  Shape info   : {len(self.shape_info)} annotated tensors",
            f"  Op types ({len(op_counts)}):",
        ]
        for op, cnt in op_counts.most_common():
            lines.append(f"    {op:<24} {cnt:>5}")
        return "\n".join(lines)
