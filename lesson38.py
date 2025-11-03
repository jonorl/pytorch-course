import torch
from torch import nn ##neural network
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_device('cuda')

# Linear regression: Y = a + bX
weight = 0.7 # b
bias = 0.3 # a

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) #unsqueeze just to remove the extra bracket
y = weight * X + bias

# Create a linear regression model class

class LinearRegressionModel(nn.Module): #inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias == nn.Parameter(torch.randn(1,
                                               requires_grad=True,                                               requires_grad=True,
                                               dtype=torch.float))
        
# Forward method to define the computation in the model
# 
def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data
    return self.weights * x + self.bias # this is the linear regression formula
 
## What our model does:

# 1. Start with random values (weight & bias)
# 2. Look at training data and adjust the random values to get as close as possible to
# the ideal values we used to create the data

## How does it do it

# Through 2 main algorithms
# 1. Gradient descent
# 2. Backpropagation 