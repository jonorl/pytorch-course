import torch
from torch import nn ##neural network
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Device agnostic set up
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

## Loading a pytorch model (state dict)

# Create PATHs

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "model_0.pt" # Replace with whatever model you want to load
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Create data

# Linear regression: Y = a + bX
weight = 0.7 # b
bias = 0.3 # a

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) #unsqueeze just to remove the extra bracket
y = weight * X + bias

# Split data between training and testing:

train_split = int(0.8 * len(X))

X_Train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Create the Linear Regression Model from nn.Module:

class LinearRegressionModel(nn.Module): #inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype=torch.float))
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data
        return self.weights * x + self.bias # this is the linear regression formula
    
# Create the new model from the class which will include random parameters:

loaded_model_0 = LinearRegressionModel()

print(f"Initial random dict values: {loaded_model_0.state_dict()}")
loaded_model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))
print(f"Loaded dict values: {loaded_model_0.state_dict()}")

# Make predictions with the loaded model

loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

print(f"Loaded model predictions: {loaded_model_preds}")