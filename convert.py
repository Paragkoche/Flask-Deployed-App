import torch
import torch.onnx
from CNN import CNN  # Adjust this import according to your actual module

# Load the pre-trained model
model = CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

# Create dummy input with the same shape as the input your model expects
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model
torch.onnx.export(model, dummy_input, "plant_disease_model.onnx")
