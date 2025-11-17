import torch
from torch import nn ##neural network
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

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
n_samples = 1000

# Create circles

X, y = make_circles(n_samples, 
                    noise=0.03,
                    random_state=42)

# print(f"First five samples of X: {X[:5]}")
# print(f"First five samples of y: {y[:5]}")

# Uncomment to visualise
# Make a DataFrame of circle data
# This show either an external circle (0) or internal circle (1)

# circles = pd.DataFrame({"X1": X[:, 0],
#                         "X2": X[:, 1],
#                         "label": y})

# plt.figure(figsize=(8, 6))
# plt.scatter(x=X[:, 0],
#             y=X[:, 1],
#             c=y,
#             cmap=plt.cm.RdYlBu)
# plt.title("Circle Dataset")
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.show()

## Turn data from numpy into tensors

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split data into training and testing

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, #20% of data will be test
                                                    random_state=42)

# print(len(X_train), len(X_test), len(y_train), len(y_test))

## Building a model

# Subclass nn.module

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # Create nn.linear layers capable of handling our data
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        # out features from previous layer needs to match in features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) 
    
    # Define a forward method that outlines forward pass
    def forward(self, x):
        return self.layer_2(self.layer_1(x)) # x goes into layer 1 and then goes into layer 2
    
    # x -> layer_1 -> layer_2 -> output

    # initiate an instance of the model class and send it to the device

model_0 = CircleModelV0().to(device)
print(model_0.state_dict())

# Let's replicate the model above using nn.Sequential()
# This is 10 times easier than subclassing, but it's not flexible
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)).to(device)

print("Sequential model: ", model_0)

# Make predictions
with torch.inference_mode():
    untrained_pred = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_pred)}, Shape: {untrained_pred.shape}")
print(f"Length of test samples: {len(X_test)}, Shape: {X_test.shape}")
print(f"\nFirst 10 predictions:\n{torch.round(untrained_pred[:10])}")
print(f"\nFirst 10 labels:\n{y_test[:10]}")

## Loss function

# We'll use BCE (binary cross entropy) with Logits loss

loss_fn = nn.BCEWithLogitsLoss() # this has the sigmoid activation functions built-in
optimiser = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

# Calculate accuracy (out of 100 examples, how many did the model get right)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item() # How many of y_true match y_pred
    acc = (correct / len(y_pred)) * 100
    return acc


## Train model

# Going from raw logits (model output) -> prediction probs -> prediction labels
# We can convert the logtis by passing them torugh some kind of activation function
# (e.g. simoid for binary classification  and softmax for multiclass)

# Forward pass
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
    print("y_logits\n",y_logits) # this shows tensors, not categories

# use the sigmoid activation function to turn logits into predictions
y_pred_probs = torch.sigmoid(y_logits)
print("y_pred_probs\n",y_pred_probs) # still tensors
print("rounded:", torch.round(y_pred_probs)) # the output we want

# Find predicted labels
y_preds = torch.round(y_pred_probs) # labels

# Logits -> pred probs -> pred labels
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# Check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

print("finally...\n", y_preds.squeeze()) # Compare both random values (untrained)

