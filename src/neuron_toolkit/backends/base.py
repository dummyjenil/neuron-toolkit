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

    def to_dict(self) -> dict[str, object]:
        """Export the model graph structure as a Python dictionary without raw weights."""
        from neuron_toolkit.exporter import export_graph_dict

        return export_graph_dict(self)

    def to_json(self, path: str | None = None, indent: int = 2) -> str | None:
        """Export the model graph structure as a JSON string or save to a file path."""
        from neuron_toolkit.exporter import export_graph_json

        return export_graph_json(self, path=path, indent=indent)

    def save_safetensors(self, path: str, metadata: dict[str, str] | None = None) -> None:
        """Save model weight and parameter tensors into a .safetensors file."""
        from neuron_toolkit.exporter import export_safetensors

        export_safetensors(self, path, metadata=metadata)

    def export_safetensors(self, path: str, metadata: dict[str, str] | None = None) -> None:
        """Save model weight and parameter tensors into a .safetensors file (alias)."""
        self.save_safetensors(path, metadata=metadata)

    def replace_weights(
        self,
        weights_dict: dict[str, object],
        *,
        strict: bool = False,
    ) -> dict[str, list[str]]:
        """Replace model weights and initializers in-place from a dictionary."""
        tensor_map = getattr(self, "tensor_map", {})
        model_keys = set(tensor_map.keys())
        provided_keys = set(weights_dict.keys())

        missing_keys = sorted(model_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - model_keys)

        if strict:
            if missing_keys:
                msg = f"Missing key(s) in weights: {missing_keys}"
                raise ValueError(msg)
            if unexpected_keys:
                msg = f"Unexpected key(s) in weights: {unexpected_keys}"
                raise ValueError(msg)

        if isinstance(tensor_map, dict):
            for k, v in weights_dict.items():
                if k in model_keys:
                    tensor_map[k] = v

        return {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }

    def load_safetensors(
        self,
        path: str,
        *,
        strict: bool = False,
    ) -> dict[str, list[str]]:
        """Load weights from a .safetensors file and replace in-place."""
        from neuron_toolkit.exporter import load_safetensors

        weights = load_safetensors(path)
        return self.replace_weights(weights, strict=strict)


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
