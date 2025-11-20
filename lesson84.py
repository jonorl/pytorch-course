## Multi class clasification

import torch
from torch import nn ##neural network
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sklearn
from sklearn.datasets import make_blobs
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

# Hyperparameters

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES, # 2 dimensions (x and y)
                            centers=NUM_CLASSES,
                            cluster_std=1.5, # How far they are from the cluster center
                            random_state=RANDOM_SEED)

# Turn data into tensors - we need LongTensor this time
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# Split into train and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                    y_blob,
                                                    test_size=0.2, #20% of data will be test
                                                    random_state=RANDOM_SEED)

X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

# Plot

# plt.figure(figsize=(10,7))
# plt.scatter(X_blob[:,0],X_blob[:,1],c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()

## Build multi-class classification model

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """ Initialises multi-class classification model
        
        Args:
            input_features (int): number of input features to the model
            output_features (int): Number of output features (number of output classes)
            hidden_units (int): Number of hiddent units between laters, default 8
            
            Returns:
            
            Example:
            """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            # nn.ReLU(), # ReLU is optional as you can slice this with lines
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(), # ReLU is optional as you can slice this with lines
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)

model_4 = BlobModel(input_features=2, 
                    output_features=4, 
                    hidden_units=8).to(device)

# Create loss function - CrossEntropyLoss as opposed to Binary Cross entropy for multi-class

loss_fn = nn.CrossEntropyLoss()

# Create optimiser

optimiser = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1)

## in order to evaluate and train and tets our model, we need to convert the model output
## (logits) to prediction probabilities and then labels

model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test.to(device))
    
    # convert logit outputs to prediction probabilities
    y_pred_probs = torch.softmax(y_logits, dim=1) # this returns probability % for each index

y_preds = torch.argmax(y_pred_probs, dim=1)

# print(y_preds[:20])

## Build a training loop

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

epochs = 101

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item() # How many of y_true match y_pred
    acc = (correct / len(y_pred)) * 100
    return acc

for epoch in range(epochs):
    model_4.train()
    y_logits = model_4(X_blob_train).squeeze()
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)
    
    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    ## Testing
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test,
                        y_pred=test_preds)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}% | Test Loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")

    # Make predictions
    model_4.eval()
    with torch.inference_mode():
        y_logits = model_4(X_blob_test)
        y_pred_probs = torch.softmax(y_logits, dim=1).argmax(dim=1)
        y_preds = torch.argmax(y_logits, dim=1)

# For comparison:

print("y preds", y_preds[:10])
print("y blob test", y_blob_test[:10])
    
# Plot

plt.figure(figsize=(10,7))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
plt.show()