import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# torch.set_default_device('cuda') # leave this one to default device to CUDA but comment if using numpy or matplotlib

# Linear regression formula with known parameters
# Linear regression: Y = a + bX

weight = 0.7 # b
bias = 0.3 # a

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) #unsqueeze just to remove the extra bracket
y = weight * X + bias

print("X[:10]", X[:10]) # first 10 values
print("y[:10]", y[:10]) # first 10 values
print("len(X)", len(X))
print("len(y)", len(y)) 

## Splitting data into training and test sets

# The "training set" is like the course material ~60-80%
# The "validation set" is like the practice exam (used often but not always) ~10-20%
# the "test set" is the final exam ~10-20%

# you tune the training set until you get better results on the validation test

## Create a train/test split
train_split = int(0.8 * len(X))
print("train_split: ", train_split)

# Create test sets for input (X) and result (y)
X_Train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_Train), len(y_train), len(X_test), len(y_test))

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

plot_predictions()
plt.title("Model Predictions vs. Actual Data")
plt.show()