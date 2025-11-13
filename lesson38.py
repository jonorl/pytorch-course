import torch
from torch import nn ##neural network
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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


### Uncomment for visualisation graph (needs CPU)

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

    # 1. Plot training data
    plt.scatter(train_data_np, train_labels_np, c="b", s=4, label="Training Data")

    # 2. Plot testing data (ground truth)
    plt.scatter(test_data_np, test_labels_np, c="g", s=4, label="Testing Data (Actual)")

    # 3. Plot predictions
    if predictions is not None:
        predictions_np = to_cpu_numpy(predictions)
        plt.scatter(test_data_np, predictions_np, c="r", s=4, label="Predictions")

plot_predictions_cpu(train_data=X_Train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test,
                     predictions=y_preds)

plt.legend(prop={"size": 14})
plt.title("Model Predictions vs. Actual Data")
plt.show()

# print("list(model_0.parameters()) : ", list(model_0.parameters()))

# print("model_0.state_dict() : ", model_0.state_dict())

### Video: 5:40:48 - We created a model, thrown a random value, compared that against
### the actual weight and result and it's far off, so now the next step is to optimise
### the results to get it closer to where we want it to be.

## Train Model

# The model is currently bad at predicting right now, so we need to train it. To measure
# how poor/wrong is performing we need to use a loss function (also called cost function or criterion)

## Things we need to train:

# Loss function
# Optimiser: Once the loss is established, it adjusts the parameters (e.g. weights & bias)
# Training Loop
# Testing loop

# The Mean Absolute error is the avg/mean distance between the current status of the model y axis (red)
# and the prediction (green). In torch this is torch.nn.L1Loss

# Setting up the loss function

loss_fn = nn.L1Loss() # Using L1 but there are many

# Setup optimser - using SRD stochastic random descent which
# is gradient descent from random (stochastic) values.

optmiser = torch.optim.SGD(params = model_0.parameters(), #Choose the model you want to optimise 
                           lr=0.01) # Learning rate

# The optimiser applies random values to params and checks the loss
# value, if it gets "better" then it keeps adjusting in that direction

## Building a training loop

# an epoch is a single loop
torch.manual_seed(42)
epochs = 250

# track different values
epoch_count = []
loss_values = []
test_loss_values = []

# loop through data
for epoch in range(epochs):
    # 0. set the model to training mode
    model_0.train() # sets everything to gradient descent to get gradient 0, which is loss 0

    # 1. forward pass (move forward from input to output on the neural network)
    y_pred = model_0(X_Train)

    # 2. calculate the loss (avg diff between prediction and ideal values)
    loss = loss_fn(y_pred, y_train)
    # print(f"loss: {loss}")

    # 3. Optimiser zero grad - like resetting the optimiser value
    optmiser.zero_grad()

    # 4. Back Propagation
    loss.backward()

    # 5. Gradient Descent
    optmiser.step()

    ## Testing (which is not the same as training!)
    model_0.eval() # turns off settings not needed for evaluation
    # print("model_0.state_dict() : ", model_0.state_dict())

    model_0.eval() # turns off gradient tracking

    with torch.inference_mode(): # turns off graident tracking + other stuff
        # 1. Forward pass
        test_pred = model_0(X_test)
        #2. Calculate loss
        test_loss = loss_fn(test_pred,y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} , Loss {loss} , test loss: {test_loss}")
        print("model_0.state_dict(): ", model_0.state_dict())



with torch.inference_mode():
    y_preds_new = model_0(X_test)
plot_predictions_cpu(train_data=X_Train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test,
                     predictions=y_preds_new)
plt.title("New Model Predictions vs. Actual Data")
plt.show() 

## Plot the loss curves
plt.plot(epoch_count, np.array(torch.tensor(loss_values).cpu().numpy()), label="train loss")
plt.plot(epoch_count, np.array(torch.tensor(test_loss_values).cpu().numpy()), label="test loss")
plt.title("training and test loss curves")
plt.ylabel("loss")
plt.xlabel("Epochs")
plt.legend 

# How to save

# 1. Create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Crate model save path
MODEL_NAME = "model_0.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to:{MODEL_SAVE_PATH}")
torch.save(model_0.state_dict(), MODEL_SAVE_PATH)