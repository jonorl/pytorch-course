import torch
from torch import nn ##neural network
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from helper_functions import plot_predictions, plot_decision_boundary

## Setup

# Device agnostic set up
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

# Setup of plotting on CPU

def plot_predictions_cpu(train_data, 
                         train_labels, 
                         test_data, 
                         test_labels,
                         predictions=None):
    
    # Helper function to safely convert tensor to numpy array on CPU
    def to_cpu_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy()
        return data

    # Convert all input data to CPU NumPy arrays
    train_data_np = to_cpu_numpy(train_data)
    train_labels_np = to_cpu_numpy(train_labels)
    test_data_np = to_cpu_numpy(test_data)
    test_labels_np = to_cpu_numpy(test_labels)

    plt.figure(figsize=(10, 7))
    plt.scatter(train_data_np, train_labels_np, c="b", s=4, label="Training Data")
    plt.scatter(test_data_np, test_labels_np, c="g", s=4, label="Testing Data (Actual)")
    if predictions is not None:
        predictions_np = to_cpu_numpy(predictions)
        plt.scatter(test_data_np, predictions_np, c="r", s=4, label="Predictions")

# Create PATHs

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "model_1.pt" # Replace with whatever model you want to load
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

## End of setup

## Data creation

# Create data using linear regression formula of y = weight * X + bias
weight = 0.7
bias = 0.3

# Create random range values
start = 0
end = 1
step = 0.02

X_regression = torch.arange(start, end, step).unsqueeze(dim=1) #without unsqueeze errors may pop
y_regression = weight * X_regression + bias # Linear regression formula (without epsilon)

# Split data into training and testing

# Split data - 40 for training and 10 for testing
train_split = int(0.8 * len(X_regression)) # let's grab 80% of X
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split] # Make X/Y train data from the beginning up to 80%
X_test_regression, y_test_regression = X_regression[train_split:], y_regression [train_split:] # Make X/Y train data from the 80% to the end

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=1, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
    
    def forward(self, x):
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)

        # Or much simpler:
        return self.layer_3(self.layer_2(self.layer_1(x)))
    
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10,),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)).to(device)
print(model_2.state_dict())

loss_fn = nn.L1Loss()
optimiser = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.01)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000

X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item() # How many of y_true match y_pred
    acc = (correct / len(y_pred)) * 100
    return acc

for epoch in range(epochs):
    # model_2.train()
    y_pred = model_2(X_train_regression)

    loss = loss_fn(y_pred, y_train_regression)
    
    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    ## Testing

    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)

        test_loss = loss_fn(test_pred,
                        y_test_regression)
    
    # Print what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")


## Turn model into evaluation mode
model_2.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model_2(X_test_regression)

## Uncomment to see graph with predictions nearly matching test values
plot_predictions_cpu(train_data=X_train_regression, 
                     train_labels=y_train_regression, 
                     test_data=X_test_regression, 
                     test_labels=y_test_regression,
                     predictions=test_pred)
plt.title("New Model Predictions vs. Actual Data")
plt.show() 
