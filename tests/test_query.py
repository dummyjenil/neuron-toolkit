from neuron_toolkit.parser import ONNXParser


def test_query_filters(complex_model):
    parser = ONNXParser(complex_model)
    q = parser.find()

    # op_type
    assert q.find_by_op_type("Relu").count() == 1
    assert q.find_by_op_type("Add").count() == 1

    # name
    assert q.find_by_name("n1", exact=True).count() == 1
    assert q.find_by_name("n").count() == 6

    # tensor
    assert (
        q.find_by_tensor("relu_out").count() == 2
    )  # Produces by Relu, consumed by Add

    # attribute (Relu doesn't have attributes, let's use a model with attributes if needed, but we can test missing)
    assert q.find_by_attribute("non_existent").count() == 0


def test_query_traversal(complex_model):
    parser = ONNXParser(complex_model)
    q = parser.find()

    relu_node = q.find_by_name("n1", exact=True)
    add_node = q.find_by_name("n2", exact=True)

    # children
    assert relu_node.children().single_node.name == "n2"

    # parents
    assert add_node.parents().single_node.name == "n1"

    # descendants
    desc = relu_node.descendants()
    assert desc.count() == 4  # Add, Mul, Sigmoid, Tanh
    assert set(n.name for n in desc) == {"n2", "n3", "n4", "n5"}

    # ancestors
    anc = q.find_by_name("n4", exact=True).ancestors()
    assert anc.count() == 4  # Mul, Add, Relu, Identity
    assert set(n.name for n in anc) == {"n0", "n1", "n2", "n3"}


def test_query_set_ops(complex_model):
    parser = ONNXParser(complex_model)
    q = parser.find()

    q1 = q.find_by_op_type("Relu")
    q2 = q.find_by_op_type("Add")

    union = q1 | q2
    assert union.count() == 2

    inter = union & q1
    assert inter.count() == 1
    assert inter.single_node.op_type == "Relu"

    diff = union - q1
    assert diff.count() == 1
    assert diff.single_node.op_type == "Add"


def test_query_topological_sort(complex_model):
    parser = ONNXParser(complex_model)
    q = parser.find()

    sorted_q = q.topological_sort()
    assert sorted_q.count() == 6
    assert sorted_q.is_topologically_sorted()

    # Reverse it and check
    rev_nodes = list(reversed(sorted_q.nodes))
    rev_q = q._clone(rev_nodes)
    assert not rev_q.is_topologically_sorted()
