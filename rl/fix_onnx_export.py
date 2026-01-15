"""
Fix ONNX Export to include tanh activation.
This ensures the model outputs are in [-1, 1] range, matching training behavior.
"""
import torch
import torch.nn as nn

# Import the model architecture
from falconrl3 import ActorCritic, FalconEnv

def export_to_onnx_with_tanh():
    """Export the actor network with tanh activation included."""
    # 1. Config
    MODEL_PATH = "./falcon_ppo_best.pth"
    ONNX_PATH = "./falcon_ppo_best.onnx"
    
    # 2. Setup Environment & Model
    env = FalconEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    device = torch.device("cpu")
    model = ActorCritic(state_dim, action_dim)
    
    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Create a wrapper that includes tanh
    class ActorWithTanh(nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor
        
        def forward(self, x):
            # Apply actor network to get mean
            mean = self.actor(x)
            # Apply tanh to get action in [-1, 1]
            return torch.tanh(mean)
    
    actor_with_tanh = ActorWithTanh(model.actor)
    actor_with_tanh.eval()
    
    # 4. Create Dummy Input
    dummy_input = torch.randn(1, state_dim, dtype=torch.float32, device=device)
    
    print(f"Exporting model with input shape: {dummy_input.shape}")
    print("Model will output actions in [-1, 1] range (tanh applied)")
    
    # 5. Export
    torch.onnx.export(
        actor_with_tanh,           # The wrapper with tanh
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input.1'],    # Match the input name used in JS
        output_names=['output.1'],  # Match the output name used in JS
        dynamic_axes={
            'input.1': {0: 'batch_size'},
            'output.1': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Success! Model exported to: {ONNX_PATH}")
    print("Note: Model outputs are now in [-1, 1] range (tanh already applied)")

if __name__ == "__main__":
    export_to_onnx_with_tanh()
