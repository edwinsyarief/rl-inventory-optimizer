import os
import tempfile
import unittest

import onnx
import torch

from export_to_onnx import DQN, export_model


class TestExportToOnnx(unittest.TestCase):
    def setUp(self):
        self.state_size = 2
        self.action_size = 51
        self.hidden_size = 128
        self.model = DQN(self.state_size, self.action_size, self.hidden_size)
        
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "test_model.pth")
        torch.save(self.model.state_dict(), self.model_path)
        
        self.onnx_path = os.path.join(self.temp_dir.name, "test_model.onnx")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_model(self):
        class Args:
            model_path = self.model_path
            onnx_path = self.onnx_path
            hidden_size = self.hidden_size
        
        export_model(Args())
        self.assertTrue(os.path.exists(self.onnx_path))
        
        onnx_model = onnx.load(self.onnx_path)
        onnx.checker.check_model(onnx_model)  # Raises exception if invalid

if __name__ == '__main__':
    unittest.main(verbosity=2)