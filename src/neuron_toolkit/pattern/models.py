from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from neuron_toolkit._types import TensorMap

if TYPE_CHECKING:
    from neuron_toolkit.query import NeuronQuery


class MatchResult(BaseModel):
    """Result of a pattern match."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: object
    end: object
    nodes: list[object] = Field(default_factory=list)
    bindings: dict[str, object] = Field(default_factory=dict)
    tensor_map: TensorMap = Field(default_factory=dict, exclude=True, repr=False)

    def as_query(self) -> NeuronQuery:
        """Return the matched subgraph as an NeuronQuery."""
        from neuron_toolkit.query import NeuronQuery  # noqa: PLC0415

        return NeuronQuery(list(self.nodes), self.tensor_map, list(self.nodes))

    @property
    def query(self) -> NeuronQuery:
        """Return the matched subgraph as an NeuronQuery."""
        return self.as_query()

    @property
    def subgraph(self) -> NeuronQuery:
        """Alias for query."""
        return self.as_query()
