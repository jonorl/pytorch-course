### Convoluted Neural Network

import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path 
from torchmetrics import Accuracy
from timeit import default_timer as timer
from tqdm.auto import tqdm

### SETUP

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True) # Create models directory if it doesn't exist

MODEL_NAME_0 = "fashion_mnist_model_0.pth"
MODEL_SAVE_PATH_0 = MODEL_PATH / MODEL_NAME_0

MODEL_NAME_1 = "fashion_mnist_model_1.pth"
MODEL_SAVE_PATH_1 = MODEL_PATH / MODEL_NAME_1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

### END OF SETUP

## Helper functions

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """ Prints difference between start and end time."""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

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
            "\nmodel_loss": round(loss.item(), 4),
            "\nmodel_acc": str(round(acc.item() * 100, 2)) + "%"}

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

## Create a Convolutional Neural Network
class FashionMNISTModelV2(nn.Module):
    """
    Model architecturae that replicate the TinyVGG
    model from CNN explainer website
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            # Create a conv layer
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE-1) # loops through a 2x2 matrix and returns the highest value
        )
        self.conv_block_2 = nn.Sequential(
            # Create a conv layer
            nn.Conv2d(in_channels=hidden_units, # the output of the conv_block_1
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE-1) # loops through a 2x2 matrix and returns the highest value
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*0, # multiply by ???
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        print(x.shape)
        x = self.conv_block_2(x)
        print(x.shape)
        x = self.classifier(x)
        return x

## Datasets

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

# Set Hyperparameters
EPOCHS = 3
BATCH_SIZE = 32
NUM_CLASSES = len(class_names)
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1

# Set others

accuracy_fn = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(device)

# Turn datasets into iterables (batches)

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dataloader))

## Models

class FashionMNISTModelV0(nn.Module):
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

## Loading models

# Initiate model instances
loaded_model_0 = FashionMNISTModelV0(input_shape=784,
                                      hidden_units=10,
                                      output_shape=len(class_names))

loaded_model_1 = FashionMNISTModelV1(input_shape=784,
                                     hidden_units=10,
                                     output_shape=len(class_names)).to(device)

# Load the state dictionary

print(f"Loading model 0 state dict from: {MODEL_SAVE_PATH_0}")
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH_0))

# Load state dict for model_1
print(f"Loading model 1 state dict from: {MODEL_SAVE_PATH_1}")
loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH_1))

## Run the models

# Model 0

torch.manual_seed(42)

# Measure time
train_time_start_on_gpu = timer()

# Create optimisation and eval loop using train_step() and test_step()

for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}\n-----------")
    train_step(model=loaded_model_0,
               data_loader=train_dataloader,
               loss_fn=nn.CrossEntropyLoss(),
               optimiser=torch.optim.SGD(params=loaded_model_0.parameters(), # Same as usual
                            lr=0.1) ,
               accuracy_fn=accuracy_fn,
               device='cpu')
    
    test_step(model=loaded_model_0,
              data_loader=test_dataloader,
              loss_fn=nn.CrossEntropyLoss(),
              accuracy_fn=accuracy_fn,
              device='cpu')
    
train_time_end_on_gpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device='cpu')
print("total train Time:", total_train_time_model_0)

# Get Model_0 results dictionary

model_0_results = eval_model(model=loaded_model_0,
                             data_loader=test_dataloader,
                             loss_fn=nn.CrossEntropyLoss(),
                             accuracy_fn=accuracy_fn,
                             device='cpu')

# Model 1

torch.manual_seed(42)

# Measure time
train_time_start_on_gpu = timer()

# Create optimisation and eval loop using train_step() and test_step()

for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}\n-----------")
    train_step(model=loaded_model_1,
               data_loader=train_dataloader,
               loss_fn=nn.CrossEntropyLoss(),
               optimiser=torch.optim.SGD(params=loaded_model_1.parameters(), # Same as usual
                            lr=0.1) ,
               accuracy_fn=accuracy_fn,
               device=device)
    
    test_step(model=loaded_model_1,
              data_loader=test_dataloader,
              loss_fn=nn.CrossEntropyLoss(),
              accuracy_fn=accuracy_fn,
              device=device)
    
train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)
print("total train Time:", total_train_time_model_1)

# Get Model_1 results dictionary

model_1_results = eval_model(model=loaded_model_1,
                             data_loader=test_dataloader,
                             loss_fn=nn.CrossEntropyLoss(),
                             accuracy_fn=accuracy_fn,
                             device=device)


# Model 2 - CNN

torch.manual_seed(42)

model_2 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=NUM_CLASSES).to(device)

## Compare results

print(model_0_results)
print(model_1_results)