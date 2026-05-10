import unittest
from unittest.mock import MagicMock

from neuron_toolkit.backends.base import BaseParser, BaseRewriter


class TestComprehensive(unittest.TestCase):
    """Comprehensive test for library-wide parity and abstraction."""

    def test_onnx_backend_interface(self):
        """Test that ONNX backend follows the BaseParser/BaseRewriter interface."""
        # Use a mock or a small real ONNX model if possible, but here we focus on interface
        from unittest.mock import patch

        from neuron_toolkit.backends.onnx.parser import ONNXParser

        with (
            patch("onnx.load") as mock_load,
            patch("onnx.shape_inference.infer_shapes") as mock_infer,
        ):
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            mock_infer.return_value = mock_model

            parser = ONNXParser("dummy.onnx", infer_shapes=True)
            self.assertIsInstance(parser, BaseParser)

        rewriter = parser.rewriter()
        self.assertIsInstance(rewriter, BaseRewriter)

    def test_tflite_backend_interface(self):
        """Test that TFLite backend follows the BaseParser/BaseRewriter interface."""
        from neuron_toolkit.backends.tflite.parser import TFLiteParser

        # We can't easily mock TFLite flatbuffer here without 'tflite' package
        # but we can check if it inherits correctly.
        # We'll just check the class definition if we can't instantiate it.
        self.assertTrue(issubclass(TFLiteParser, BaseParser))

        from neuron_toolkit.backends.tflite.rewriter import TFLiteRewriter

        self.assertTrue(issubclass(TFLiteRewriter, BaseRewriter))

    def test_query_engine_agnosticism(self):
        """Test that NeuronQuery works with a mock backend parser."""
        mock_backend = MagicMock(spec=BaseParser)
        mock_node = MagicMock()
        mock_node.op_type = "Relu"
        mock_node.name = "relu1"
        mock_node.input = ["input"]
        mock_node.output = ["output"]

        mock_backend.nodes = [mock_node]
        mock_backend.tensor_map = {}
        mock_backend.shape_info = {"output": (4, "float32")}
        mock_backend.get_node_attrs.return_value = {"alpha": 0.0}

        from neuron_toolkit.query import NeuronQuery

        q = NeuronQuery([mock_node], {}, [mock_node], backend=mock_backend)

        # Test op filter
        self.assertEqual(len(q.op("Relu")), 1)
        self.assertEqual(len(q.op("Conv")), 0)

        # Test attribute access (should call backend.get_node_attrs)
        results = q.attr("alpha", 0.0)
        self.assertEqual(len(results), 1)
        mock_backend.get_node_attrs.assert_called_with(mock_node)

    def test_rewriter_delegation(self):
        """Test that NeuronRewriter delegates to backend rewriter."""
        mock_impl = MagicMock(spec=BaseRewriter)
        from neuron_toolkit.rewriter import NeuronRewriter

        rewriter = NeuronRewriter(mock_impl)

        # Test delegation
        rewriter.delete([MagicMock()])
        mock_impl.delete.assert_called()

        rewriter.build(output_path="test.onnx")
        mock_impl.build.assert_called_with(output_path="test.onnx")


if __name__ == "__main__":
    unittest.main()
