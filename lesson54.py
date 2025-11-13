import torch
from torch import nn ##neural network
import matplotlib.pyplot as plt
from pathlib import Path

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
weight = 0.9
bias = 0.2

# Create random range values
start = 0
end = 1
step = 0.02

# Create x and y (features and labels)
# Create X as an array of 50 tensors with the variables above
X = torch.arange(start, end, step).unsqueeze(dim=1) #without unsqueeze errors may pop
y = weight * X + bias

# Split data - 40 for training and 10 for testing
train_split = int(0.8 * len(X)) # let's grab 80% of X
X_train, y_train = X[:train_split], y[:train_split] # Make X/Y train data from the beginning up to 80%
X_test, y_test = X[train_split:], y [train_split:] # Make X/Y train data from the 80% to the end

# Uncomment to plot the data
# plot_predictions_cpu(X_train, y_train, X_test, y_test)
# plt.title("New Model Predictions vs. Actual Data")
# plt.show()

## Building a linear model

# Subclass nn.module

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # user nn.Linear() for creating the model parameters as opposed to the manual way
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
    # The foward method now includes linear_layer which is a simplified version of writing the whole thing manually
    def forward(self, x: torch.Tensor) -> torch.Tensor: # declare that i/o should be a tensor (kinda like TS) 
        return self.linear_layer(x)
    
# set the manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()

# Explicitly move the model to the target device
model_1 = LinearRegressionModelV2().to(device)

# Show the model and state dict
# print(f"model_1: {model_1}")
# print(f"model_1.state_dict(): {model_1.state_dict()}")

## Train the model

# Set up loss function
loss_fn = nn.L1Loss() # same as MAE

# Set up optimiser
optimiser = torch.optim.SGD(params=model_1.parameters(), # Pass parameters from the model
                            lr=0.01) # Learning rate (how quickly to take each step)

# Write a training loop
torch.manual_seed(42)
epochs = 181

for epoch in range(epochs):
    # Set it to train (as opposed to test)
    model_1.train()

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad (start from fresh / don't cummulate)
    optimiser.zero_grad()

    # 4. Perform back propagation
    loss.backward()

    # 5. Optimiser step
    optimiser.step()

    ## Testing

    # Set it to eval and inference mode to avoid unnecessary training stuff
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test) # pass the X_test data to our prediction model
        test_loss = loss_fn(test_pred, y_test) # calculate the test loss

        # Print out what's happening
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}, Test loss:{test_loss}")

# This prints out weight 0.8920 and bias 0.2027
print(f"model_1.state_dict: {model_1.state_dict()}")

## Turn model into evaluation mode
model_1.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model_1(X_test)

## Uncomment to see graph with predictions nearly matching test values
# plot_predictions_cpu(train_data=X_train, 
#                      train_labels=y_train, 
#                      test_data=X_test, 
#                      test_labels=y_test,
#                      predictions=test_preds)
# plt.title("New Model Predictions vs. Actual Data")
# plt.show() 

## Saving and loading a training model

# Saving
print(f"Saving model to:{MODEL_SAVE_PATH}")
torch.save(model_1.state_dict(), MODEL_SAVE_PATH)

# Loading

loaded_model_1 = LinearRegressionModelV2() # Create new instance of linear regression v2
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

loaded_model_1.to(device) # load it to gpu

# Evaluate loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)
    print(f"Check for equality: {loaded_model_1_preds == y_preds}")