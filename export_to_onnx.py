import argparse
import logging

import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DQN(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

def export_model(args):
    try:
        # Load model
        state_size = 2  # Inventory state size: [inventory, forecast]
        action_size = 51  # 0 to 50
        model = DQN(state_size, action_size, args.hidden_size)
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        model.eval()
        
        # Dummy input for export (batch size 1, but dynamic)
        dummy_input = torch.randn(1, state_size)
        
        # Export to ONNX with optimizations
        torch.onnx.export(
            model,
            dummy_input,
            args.onnx_path,
            export_params=True,
            opset_version=13,  # Higher for performance
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logger.info(f"Model exported to ONNX: {args.onnx_path}")
    
    except Exception as e:
        logger.error(f"Error during export: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch Model to ONNX")
    parser.add_argument("--model_path", type=str, default="dqn_inventory_agent.pth", help="Path to trained PyTorch model")
    parser.add_argument("--onnx_path", type=str, default="dqn_inventory_agent.onnx", help="Path to save ONNX model")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden layer size")
    args = parser.parse_args()
    
    export_model(args)