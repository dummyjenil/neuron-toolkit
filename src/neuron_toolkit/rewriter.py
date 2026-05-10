from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuron_toolkit.backends.base import BaseRewriter


class NeuronRewriter:
    """Base interface for model transformation engines.

    This class serves as a unified entry point for all backend-specific rewriters.
    """

    def __init__(self, backend_rewriter: BaseRewriter) -> None:
        self._impl = backend_rewriter

    def build(self, output_path: str | None = None) -> Any:
        """Apply edits and return the modified model."""
        return self._impl.build(output_path=output_path)

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else to the implementation."""
        return getattr(self._impl, name)


__all__ = ["NeuronRewriter"]
