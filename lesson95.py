### Torch Vision

import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path 
from torchmetrics import Accuracy
from timeit import default_timer as timer
from tqdm.auto import tqdm

## Getting a dataset

train_data = datasets.FashionMNIST(
    root="data", # Where to download data to
    train=True, # Do we want the training dataset?
    download=True, # Do we want to download?
    transform=ToTensor(), # How do we want to transform the data
    target_transform=None # How do we want to transform the labels/targets
    )

test_data = datasets.FashionMNIST(
    root="data", # Where to download data to
    train=False, # Do we want the training dataset?
    download=True, # Do we want to download?
    transform=ToTensor(), # How do we want to transform the data
    target_transform=None # How do we want to transform the labels/targets
    )

image, label = train_data[0]
class_names = train_data.classes
class_to_idx = train_data.class_to_idx
print(class_to_idx)
print(image.shape, label)

## Visualise data

# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(label)
# plt.axis(False)
# plt.show()

# # Plot more images
# torch.manual_seed(42)
# fig = plt.figure(figsize=(9,9))
# rows, cols = 4,4
# for i in range (1,rows*cols+1):
#     random_idx = torch.randint(0, len(train_data), size=[1]) .item()
#     print(random_idx)
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)
#     plt.show()

## Prepare Data Loader

# Break data into batches to be more efficient

BATCH_SIZE = 32
NUM_CLASSES = len(class_names)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Turn datasets into iterables (batches)

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dataloader))


# Show a sample
# torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
# plt.imshow(img.squeeze(), cmap="gray")
# plt.axis(False)
# plt.title(class_names[label])
# plt.show()
# print(img.shape)
# print("label shape: ", label.shape)

# Build model

flatten_model = nn.Flatten()

x = train_features_batch[0]

# Flatten the sample

output = flatten_model(x) # Perform forward pass

print("x", x.shape) # -> 1 channel, 28 width, 28 height
print("x", output.shape)  # -> 1 channel, single vector (768 = 28*28)

class FashrionMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)
    
# Setup model with input params

torch.manual_seed(42)

model_0 = FashrionMNISTModelV0(
    input_shape=784, # 28*28
    hidden_units=10, # how many units in the hidden layer
    output_shape=len(class_names) # One for every class
)

dummy_x = torch.rand([1, 1, 28, 28]) # dummy 1 batch of 1 color channel, 28 width, 28 height
print(model_0(dummy_x))

## Setup loss function, optimiser, and evaluation metrics

loss_fn = nn.CrossEntropyLoss() # For multiclass
optimiser = torch.optim.SGD(params=model_0.parameters(), # Same as usual
                            lr=0.1) 
accuracy_fn = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(device)

 # Create a function to time our experiments

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """ Prints difference between start and end time."""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

# start_time = timer()
# end_time = timer()
# print(print_train_time(start=start_time, end=end_time, device="gpu"))

## Training loop and training a model

# Set the seed and start the timer

torch.manual_seed(42)
train_time_start_on_cpu = timer()

# Set the number of epochs (small for faster training time)
EPOCHS = 3

# Create training and test loop

for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}\n------")
    # Training
    train_loss = 0
    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(train_dataloader): # image, label
        model_0.train()
        # Forward pass
        y_pred = model_0(X)

        # Claculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulate train loss each batch

        # optimise zero gra
        optimiser.zero_grad()

        # Loss backward
        loss.backward()

        # Optimiser step
        optimiser.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")
    
# Divide total train loss by length of dataloader
train_loss /= len(train_dataloader) # avg loss per batch of 32

test_loss, test_acc = 0, 0
model_0.eval()
with torch.inference_mode():
    for X_test, y_test in test_dataloader:
        # Forward pass
        test_pred = model_0(X_test)

        # Calculate loss (accumulatively)
        test_loss += loss_fn(test_pred, y_test)

        # Calculate accuracy
        test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1))
    
    # Calculate the test loss average per batch
    test_loss /= len(test_data)

    # Calculate the test acc average per batch
    test_acc /= len(test_dataloader)

print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc*100:.4f}%")

# Calculate training time
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                            end=train_time_end_on_cpu,
                                            device=str(next(model_0.parameters()).device))
print("Total time: ", total_train_time_model_0)

## 4. Make predictions and get model 0 results

torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device=device):
    """ returns a dictionary containing results of model predicting on data_loader"""

    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):

            # Make our data device agnostic
            X, y = X.to(device), y.to(device)

            # Make predictions
            y_pred = model(X)

            # Accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1))

        # Scale loss and acc to find avg loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__, # This only works if model was created with a class}
            "\nmodel_loss": loss.item(),
            "\nmodel_acc": acc.item()}


# Calculate model 0 results on test datasets
model_0_results = eval_model(model=model_0,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             device='cpu')

print("model_0_results: ", model_0_results)

## Setup device agnostic-code

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.set_default_device(device)

## Build model 1 using GPU and non-linearity

class FashionMNISTModelV1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x:torch.Tensor):
        return self.layer_stack(x)
    
# Create an instance of model_1

torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=784,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device) # send to GPU if available

loss_fn = nn.CrossEntropyLoss() # For multiclass
optimiser = torch.optim.SGD(params=model_1.parameters(), # Same as usual
                            lr=0.1) 

## Functionising training and evaluation/testing loops

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimiser: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    
    """ Performs a training with model trying to learn on data loader"""

    # Put model into training mode
    train_loss, train_acc = 0, 0

    # Add a loop to loop through the training batches
    model.train()
    for batch, (X,y) in enumerate(data_loader):
        
        # Put data on target device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate the loss and accuracy per batch
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulate train loss
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1)).to(device) # go from logits to pred labels

        # Optimise zero grad
        optimiser.zero_grad()

        # Loss Backward
        loss.backward()

        # Optimiser step (update the model's parameters once per batch)
        optimiser.step()

    # Divide total train loss and acc by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc*100:.2f}%\n")


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    
    """ Performs a testing loop test on model going over data_loader"""

    # Put model into training mode
    test_loss, test_acc = 0, 0

    # Put model in eval mode
    model.eval()

    # Turn on inference mode context manager
    with torch.no_grad():
        for X, y in data_loader:

            # Put data on target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred = model(X)

            # Calculate the loss and accuracy per batch
            loss = loss_fn(test_pred, y)
            test_loss += loss # accumulate train loss
            test_acc += accuracy_fn(y, test_pred.argmax(dim=1)) # go from logits to pred labels

    # Divide total train loss and acc by length of train dataloader
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc*100:.2f}%\n")

torch.manual_seed(42)

# Measure time
train_time_start_on_gpu = timer()
# Set epochs
EPOCHS = 3

# Create optimisation and eval loop using train_step() and test_step()

for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}\n-----------")
    train_step(model=model_1,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimiser=optimiser,
               accuracy_fn=accuracy_fn,
               device=device)
    
    test_step(model=model_1,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
    
train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)
print("total train Time:", total_train_time_model_1)

# Get Model_1 results dictionary

model_1_results = eval_model(model=model_1,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             device=device)

print(model_1_results)

# Save models

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True) # Create models directory if it doesn't exist

MODEL_NAME_0 = "fashion_mnist_model_0.pth"
MODEL_SAVE_PATH_0 = MODEL_PATH / MODEL_NAME_0

MODEL_NAME_1 = "fashion_mnist_model_1.pth"
MODEL_SAVE_PATH_1 = MODEL_PATH / MODEL_NAME_1

# Save model_0 state dict
print(f"Saving model 0 to: {MODEL_SAVE_PATH_0}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH_0)

# Save model_1 state dict
print(f"Saving model 1 to: {MODEL_SAVE_PATH_1}")
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH_1)