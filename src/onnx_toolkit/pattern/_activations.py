from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from onnx_toolkit.pattern.dsl import Pattern


class ActivationMixin:
    """Mixin for activation functions in the Pattern DSL."""

    @classmethod
    def relu(cls, x: Pattern) -> Pattern:
        """Create a Relu pattern."""
        return cls.op("Relu")(x)

    @classmethod
    def sigmoid(cls, x: Pattern) -> Pattern:
        """Create a Sigmoid pattern."""
        return cls.op("Sigmoid")(x)

    @classmethod
    def tanh(cls, x: Pattern) -> Pattern:
        """Create a Tanh pattern."""
        return cls.op("Tanh")(x)

    @classmethod
    def leaky_relu(cls, x: Pattern) -> Pattern:
        """Create a LeakyRelu pattern."""
        return cls.op("LeakyRelu")(x)

    @classmethod
    def elu(cls, x: Pattern) -> Pattern:
        """Create an Elu pattern."""
        return cls.op("Elu")(x)

    @classmethod
    def selu(cls, x: Pattern) -> Pattern:
        """Create a Selu pattern."""
        return cls.op("Selu")(x)

    @classmethod
    def softplus(cls, x: Pattern) -> Pattern:
        """Create a Softplus pattern."""
        return cls.op("Softplus")(x)

    @classmethod
    def softsign(cls, x: Pattern) -> Pattern:
        """Create a Softsign pattern."""
        return cls.op("Softsign")(x)

    @classmethod
    def hardsigmoid(cls, x: Pattern) -> Pattern:
        """Create a HardSigmoid pattern."""
        return cls.op("HardSigmoid")(x)

    @classmethod
    def hardswish(cls, x: Pattern) -> Pattern:
        """Create a HardSwish pattern."""
        return x * cls.hardsigmoid(x)

    @classmethod
    def silu(cls, x: Pattern) -> Pattern:
        """Create a SiLU pattern."""
        return x * cls.sigmoid(x)

    @classmethod
    def swish(cls, x: Pattern) -> Pattern:
        """Create a Swish pattern (alias for SiLU)."""
        return x * cls.sigmoid(x)

    @classmethod
    def gelu(cls, x: Pattern) -> Pattern:
        """Create a GeLU pattern (standard approximation)."""
        return x * (cls.op("Erf")(x / cls.const(1.41421356237)) + cls.const(1.0)) * cls.const(0.5)

    @classmethod
    def gelu_tanh(cls, x: Pattern) -> Pattern:
        """Create a GeLU pattern (Tanh approximation)."""
        return (
            cls.const(0.5)
            * x
            * (
                cls.const(1.0)
                + cls.tanh(
                    cls.const(0.7978845608) * (x + cls.const(0.044715) * (x**cls.const(3.0)))
                )
            )
        )

    @classmethod
    def mish(cls, x: Pattern) -> Pattern:
        """Create a Mish activation pattern."""
        return x * cls.tanh(cls.softplus(x))

    @classmethod
    def relu6(cls, x: Pattern) -> Pattern:
        """Create a ReLU6 pattern."""
        from onnx_toolkit.pattern.dsl import Pattern, _coerce

        return Pattern("Clip", [_coerce(x), cls.const(0.0), cls.const(6.0)])

    @classmethod
    def softmax(cls, x: Pattern) -> Pattern:
        """Create a Softmax pattern."""
        return cls.op("Softmax")(x)

    @classmethod
    def log_softmax(cls, x: Pattern) -> Pattern:
        """Create a LogSoftmax pattern."""
        return cls.op("LogSoftmax")(x)

    @classmethod
    def prelu(cls, x: Pattern, slope: Pattern) -> Pattern:
        """Create a PRelu pattern."""
        return cls.op("PRelu", x, slope)

    @classmethod
    def thresholded_relu(cls, x: Pattern) -> Pattern:
        """Create a ThresholdedRelu pattern."""
        return cls.op("ThresholdedRelu")(x)
