"""Mixin for activation functions in the Pattern DSL."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from neuron_toolkit.pattern.dsl import Pattern


class ActivationMixin:
    """Mixin for activation functions in the Pattern DSL."""

    @classmethod
    def relu(cls, x: Pattern) -> Pattern:
        """Create a Relu pattern."""
        return cast(Any, cls).op("Relu")(x)

    @classmethod
    def sigmoid(cls, x: Pattern) -> Pattern:
        """Create a Sigmoid pattern."""
        return cast(Any, cls).op("Sigmoid")(x)

    @classmethod
    def tanh(cls, x: Pattern) -> Pattern:
        """Create a Tanh pattern."""
        return cast(Any, cls).op("Tanh")(x)

    @classmethod
    def leaky_relu(cls, x: Pattern) -> Pattern:
        """Create a LeakyRelu pattern."""
        return cast(Any, cls).op("LeakyRelu")(x)

    @classmethod
    def elu(cls, x: Pattern) -> Pattern:
        """Create an Elu pattern."""
        return cast(Any, cls).op("Elu")(x)

    @classmethod
    def selu(cls, x: Pattern) -> Pattern:
        """Create a Selu pattern."""
        return cast(Any, cls).op("Selu")(x)

    @classmethod
    def softplus(cls, x: Pattern) -> Pattern:
        """Create a Softplus pattern."""
        return cast(Any, cls).op("Softplus")(x)

    @classmethod
    def softsign(cls, x: Pattern) -> Pattern:
        """Create a Softsign pattern."""
        return cast(Any, cls).op("Softsign")(x)

    @classmethod
    def hardsigmoid(cls, x: Pattern) -> Pattern:
        """Create a HardSigmoid pattern."""
        return cast(Any, cls).op("HardSigmoid")(x)

    @classmethod
    def hardswish(cls, x: Pattern) -> Pattern:
        """Create a HardSwish pattern."""
        return x * cast(Any, cls).hardsigmoid(x)

    @classmethod
    def silu(cls, x: Pattern) -> Pattern:
        """Create a SiLU pattern."""
        return x * cast(Any, cls).sigmoid(x)

    @classmethod
    def swish(cls, x: Pattern) -> Pattern:
        """Create a Swish pattern (alias for SiLU)."""
        return x * cast(Any, cls).sigmoid(x)

    @classmethod
    def gelu(cls, x: Pattern) -> Pattern:
        """Create a GeLU pattern (standard approximation)."""
        erf_val = cast(Any, cls).op("Erf")(
            x / cast(Any, cls).const(1.41421356237)
        )
        return x * (erf_val + cast(Any, cls).const(1.0)) * cast(Any, cls).const(0.5)

    @classmethod
    def gelu_tanh(cls, x: Pattern) -> Pattern:
        """Create a GeLU pattern (Tanh approximation)."""
        return (
            cast(Any, cls).const(0.5)
            * x
            * (
                cast(Any, cls).const(1.0)
                + cast(Any, cls).tanh(
                    cast(Any, cls).const(0.7978845608)
                    * (
                        x
                        + cast(Any, cls).const(0.044715)
                        * (x ** cast(Any, cls).const(3.0))
                    )
                )
            )
        )

    @classmethod
    def mish(cls, x: Pattern) -> Pattern:
        """Create a Mish activation pattern."""
        return x * cast(Any, cls).tanh(cast(Any, cls).softplus(x))

    @classmethod
    def relu6(cls, x: Pattern) -> Pattern:
        """Create a ReLU6 pattern."""
        from neuron_toolkit.pattern.dsl import Pattern, _coerce  # noqa: PLC0415

        return Pattern(
            "Clip",
            [_coerce(x), cast(Any, cls).const(0.0), cast(Any, cls).const(6.0)],
        )

    @classmethod
    def softmax(cls, x: Pattern) -> Pattern:
        """Create a Softmax pattern."""
        return cast(Any, cls).op("Softmax")(x)

    @classmethod
    def log_softmax(cls, x: Pattern) -> Pattern:
        """Create a LogSoftmax pattern."""
        return cast(Any, cls).op("LogSoftmax")(x)

    @classmethod
    def prelu(cls, x: Pattern, slope: Pattern) -> Pattern:
        """Create a PRelu pattern."""
        return cast(Any, cls).op("PRelu", x, slope)

    @classmethod
    def thresholded_relu(cls, x: Pattern) -> Pattern:
        """Create a ThresholdedRelu pattern."""
        return cast(Any, cls).op("ThresholdedRelu")(x)
