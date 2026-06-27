"""Base classes for model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuron_toolkit.pattern import MatchResult, Pattern
    from neuron_toolkit.query import NeuronQuery


class BaseParser(ABC):
    """Abstract base class for model parsers."""

    @abstractmethod
    def find(self) -> NeuronQuery:
        """Return a query over all nodes."""
        pass

    @abstractmethod
    def pattern_detect(
        self,
        pattern: Pattern,
        start_node: object | None = None,
        end_node: object | None = None,
    ) -> MatchResult | None:
        """Find a pattern in the model."""
        pass

    @abstractmethod
    def rewriter(self) -> BaseRewriter:
        """Return a rewriter for this model."""
        pass

    @abstractmethod
    def get_node_attrs(self, node: object) -> dict[str, object]:
        """Extract attributes from a node."""
        pass

    @abstractmethod
    def is_constant_node(self, node: object) -> bool:
        """Check if a node is a constant-provider."""
        pass

    @abstractmethod
    def get_constant_value(self, node: object) -> object | None:
        """Extract the constant value from a node."""
        pass

    @abstractmethod
    def summary(self) -> str:
        """Return a summary of the model."""
        pass


class BaseRewriter(ABC):
    """Abstract base class for model rewriters."""

    @abstractmethod
    def replace(
        self,
        nodes: Sequence[object],
        new_op: str,
        inputs: list[str],
        outputs: list[str],
        name: str | None = None,
        **attrs: object,
    ) -> BaseRewriter:
        """Replace nodes with a new op."""
        pass

    @abstractmethod
    def replace_from_result(
        self,
        result: MatchResult,
        new_op: str,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        name: str | None = None,
        **attrs: object,
    ) -> BaseRewriter:
        """Replace a matched subgraph."""
        pass

    @abstractmethod
    def delete(self, nodes: Sequence[object]) -> BaseRewriter:
        """Delete nodes."""
        pass

    @abstractmethod
    def insert_before(
        self,
        target_node: object,
        new_op: str,
        inputs: list[str],
        outputs: list[str],
        name: str | None = None,
        **attrs: object,
    ) -> BaseRewriter:
        """Insert a node before target."""
        pass

    @abstractmethod
    def reset(self) -> BaseRewriter:
        """Reset staged edits."""
        pass

    @abstractmethod
    def build(self, output_path: str | None = None) -> object:
        """Apply edits and return the new model."""
        pass
