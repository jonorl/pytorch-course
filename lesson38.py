import torch
from torch import nn ##neural network
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# torch.set_default_device('cuda')

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
        self.bias = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
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
train_split = int(0.8 * len(X))

X_Train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Create a Random Seed
torch.manual_seed(42)

# Create an instance of the model using the class
model_0 = LinearRegressionModel()

# Make predictions with model

with torch.inference_mode():
    y_preds = model_0(X_test)

print("y_preds: ", y_preds)
print("y_test: ", y_test)


def plot_predictions(train_data=X_Train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predicitons")
    
    plt.legend(prop={"size": 14})


plot_predictions(predictions=y_preds)
# plot_predictions()
plt.title("Model Predictions vs. Actual Data")
plt.show() 

### Video: 5:40:48 - We created a model, thrown a random value, compared that against
### the actual weight and result and it's far off, so now the next step is to optimise
### the results to get it closer to where we want it to be.