## Non-linear function

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

## Make 1000 samples
n_samples = 100


# Create circles

X, y = make_circles(n_samples, 
                    noise=0.03,
                    random_state=42)

## Turn data from numpy into tensors

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split data into training and testing

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, #20% of data will be test
                                                    random_state=42)

## Building a NON-linear model

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=128)
        self.layer_2 = nn.Linear(in_features=128, out_features=256)
        self.layer_3 = nn.Linear(in_features=256, out_features=1)
        # Non-linear activation function
        self.relu = nn.ReLU() # turns negative numbers to 0 and leave positive ints as they are

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
    
model_3 = CircleModelV2().to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimiser = torch.optim.SGD(params=model_3.parameters(),
                            lr=0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

epochs = 501

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item() # How many of y_true match y_pred
    acc = (correct / len(y_pred)) * 100
    return acc

for epoch in range(epochs):
    model_3.train()
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) #logits -> pred probs -> pred labels

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    ## Testing

    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

    test_loss = loss_fn(test_logits,
                        y_test)
    
    test_acc = accuracy_fn(y_true=y_test,
                           y_pred=test_pred)
    
    # Print what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f}% | Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}")

## Now the circle is displayed properly (uncomment to see):

# plt.figure(figsize=(12, 6))
# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(model_3, X_train, y_train)
# plt.show()
# plt.figure(figsize=(12, 6))
# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(model_3, X_test,y_test)
# plt.show()

## Building non-linear functions from scratch.

# Create a tensor

A = torch.arange(-10,10,1, dtype=torch.float32)

# plt.plot(A.to('cpu'))
# plt.show()
# plt.plot(torch.relu(A.to('cpu')))
# plt.show()

# Relu essentially picks the max between a given number and 0
def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0, device=x.device),x)

print (relu(A))

# plt.plot(relu(A.to('cpu')))
# plt.show()

# Now let's do the same for sigmoid

def sigmoid(x: torch.Tensor)  -> torch.Tensor:
    return 1 / (1 + torch.exp(torch.tensor(-x,device=x.device)))

print(sigmoid(A))

# plt.plot(torch.sigmoid(A.to('cpu')))
# plt.show()
# plt.plot(sigmoid(A.to('cpu')))
# plt.show()