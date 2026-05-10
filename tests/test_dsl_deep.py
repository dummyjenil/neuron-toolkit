import pytest

from neuron_toolkit.pattern.dsl import _ANY_OF, _CONST_PAT, _WILDCARD, Pattern


def test_pattern_basics():
    p = Pattern.any()
    assert p.op_type == _WILDCARD

    p = Pattern.const(5.0)
    assert p.op_type == _CONST_PAT
    assert p.value == 5.0

    p = Pattern.op("Add", Pattern.any(), Pattern.const(1))
    assert p.op_type == "Add"
    assert len(p.inputs) == 2
    assert p.inputs[0].op_type == _WILDCARD
    assert p.inputs[1].op_type == _CONST_PAT


def test_pattern_any_of():
    p1 = Pattern.op("Relu")
    p2 = Pattern.op("LeakyRelu")
    p = Pattern.any_of(p1, p2)
    assert p.op_type == _ANY_OF
    assert p._alternatives == [p1, p2]

    with pytest.raises(ValueError, match="at least 2 alternatives"):
        Pattern.any_of(p1)


def test_pattern_fluent_api():
    p = (
        Pattern.any()
        .capture("x")
        .where(alpha=0.5)
        .with_output_rank(2)
        .with_dtype("float32")
    )
    assert p._capture == "x"
    assert p._constraints == {"alpha": 0.5}
    assert p._rank == 2
    assert p._dtype == "float32"
    assert p.op_type == _WILDCARD


def test_pattern_arithmetic():
    x = Pattern.any().capture("x")

    # x + 1
    p = x + 1
    assert p.op_type == "Add"
    assert p.inputs[0] is x
    assert p.inputs[1].op_type == _CONST_PAT
    assert p.inputs[1].value == 1

    # 1 + x
    p = 1 + x
    assert p.op_type == "Add"
    assert p.inputs[0].value == 1
    assert p.inputs[1] is x

    # x * 2
    p = x * 2
    assert p.op_type == "Mul"

    # x / 2
    p = x / 2
    assert p.op_type == "Div"

    # x ** 2
    p = x**2
    assert p.op_type == "Pow"

    # -x
    p = -x
    assert p.op_type == "Neg"
    assert p.inputs[0] is x


def test_pattern_call():
    relu = Pattern.op("Relu")
    x = Pattern.any()
    p = relu(x)
    assert p.op_type == "Relu"
    assert p.inputs == [x]

    with pytest.raises(ValueError, match="already has inputs"):
        p(x)


def test_pattern_activations():
    x = Pattern.any()

    p = Pattern.relu(x)
    assert p.op_type == "Relu"

    p = Pattern.sigmoid(x)
    assert p.op_type == "Sigmoid"

    p = Pattern.hardswish(x)
    assert p.op_type == "Mul"  # x * hardsigmoid(x)

    p = Pattern.gelu(x)
    assert p.op_type == "Mul"

    p = Pattern.relu6(x)
    assert p.op_type == "Clip"
    assert p.inputs[1].value == 0.0
    assert p.inputs[2].value == 6.0


def test_pattern_mish():
    x = Pattern.any()
    p = Pattern.mish(x)
    # x * tanh(softplus(x))
    assert p.op_type == "Mul"
    assert p.inputs[0] is x
    tanh = p.inputs[1]
    assert tanh.op_type == "Tanh"
    softplus = tanh.inputs[0]
    assert softplus.op_type == "Softplus"
    assert softplus.inputs[0] is x
