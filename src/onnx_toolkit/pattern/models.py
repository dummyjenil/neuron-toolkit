from __future__ import annotations

from typing import TYPE_CHECKING

from onnx.onnx_pb import NodeProto
from pydantic import BaseModel, ConfigDict, Field

from onnx_toolkit._types import TensorMap

if TYPE_CHECKING:
    from onnx_toolkit.query import ONNXQuery


class MatchResult(BaseModel):
    """Result of a pattern match."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: NodeProto
    end: NodeProto
    nodes: list[NodeProto] = Field(default_factory=list)
    bindings: dict[str, NodeProto] = Field(default_factory=dict)
    tensor_map: TensorMap = Field(default_factory=dict, exclude=True, repr=False)

    def as_query(self) -> ONNXQuery:
        """Return the matched subgraph as an ONNXQuery."""
        from onnx_toolkit.query import ONNXQuery

        return ONNXQuery(list(self.nodes), self.tensor_map, list(self.nodes))

    @property
    def query(self) -> ONNXQuery:
        """Return the matched subgraph as an ONNXQuery."""
        return self.as_query()

    @property
    def subgraph(self) -> ONNXQuery:
        """Alias for query."""
        return self.as_query()
