import torch
import torch.nn as nn

from LMCE.networks import MLP
from LMCE.model_to_c_conversion import exportNet, nn_c_model_test

model = MLP(input_size=19, output_size=6, hidden_layers=[32])

# Set weights and biases to zero
with torch.no_grad():
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.zeros_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
            
model.save("mlp_zeroed")

# Generate the model c code
model_path = "./models/mlp_zeroed.pth"
exportNet(model_path, info="Null neural network")

# Test if the generated model gives the same outputs as the original
model.double()
nn_c_model_test(model, in_dim=19)
