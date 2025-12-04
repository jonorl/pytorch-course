## Creating a Model without augmentation

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchinfo import summary
from pathlib import Path
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from typing import Tuple, Dict, List
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torchmetrics import Accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True) # Create models directory if it doesn't exist

MODEL_NAME_0 = "pizza_steak_sushi_model_0.pth"
MODEL_SAVE_PATH_0 = MODEL_PATH / MODEL_NAME_0

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
        train_loss += loss.item() # accumulate train loss
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred) # go from logits to pred labels

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
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn=None,
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
            test_pred_logits = model(X)

            # Calculate the loss and accuracy per batch
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item() # accumulate train loss
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Divide total train loss and acc by length of train dataloader
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc*100:.2f}%\n")
    return test_loss, test_acc

def train(model:torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          accuracy_fn,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device=device
          ):
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimiser=optimizer,
                                           accuracy_fn=accuracy_fn,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                           data_loader=test_dataloader,
                                           loss_fn=loss_fn,
                                           accuracy_fn=accuracy_fn,
                                           device=device)
        
        print(f"Epoch: {epoch} | Train loss: {train_loss:4f} | Train acc: {train_acc:.4f} | test loss: {test_loss:.4f} | test acc: {test_acc:.4f}")
    
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
EPOCHS = 5


train_data_simple = datasets.ImageFolder(root=train_dir,
                                         transform=simple_transform)

test_data_simple = datasets.ImageFolder(root=test_dir,
                                         transform=simple_transform)

train_dataloader_simple = DataLoader(dataset=train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS)

class_names_simple = train_data_simple.classes

NUM_CLASSES = len(class_names_simple)
accuracy_fn = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(device)

## Create TinyVGG model

class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from CNN Explainer
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, input_size: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape, input_size, input_size)
            x = self.conv_block_1(dummy_input)
            x = self.conv_block_2(x)
            flattened_size = x.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_size,
                     out_features=output_shape )
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

INPUT_SIZE = simple_transform.transforms[0].size[0]


torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3,
                  hidden_units=16,
                  input_size=INPUT_SIZE,
                  output_shape=len(class_names_simple)).to(device)
                 
image_batch, label_batch = next(iter(train_dataloader_simple))

# print(model_0(image_batch.to(device)))
# print(summary(model_0, input_size=[BATCH_SIZE, len(class_names_simple), INPUT_SIZE, INPUT_SIZE]))

LOSS_FN = nn.CrossEntropyLoss()
OPTIMISER = torch.optim.Adam(params=model_0.parameters(),                             
                             lr=0.001)

# OPTIMISER = optimiser=torch.optim.SGD(params=model_0.parameters(), # Same as usual
#                             lr=0.1)

# Train and test model

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Measure time
train_time_start_model_0 = timer()

# Create optimisation and eval loop using train_step() and test_step()

model_0_results = train(model=model_0,
                        train_dataloader=train_dataloader_simple,
                        test_dataloader=test_dataloader_simple,
                        optimizer=OPTIMISER,
                        accuracy_fn=accuracy_fn,
                        loss_fn=LOSS_FN
                        ,epochs=EPOCHS)

# for epoch in tqdm(range(EPOCHS)):
#     print(f"Epoch: {epoch}\n-----------")
#     train_step(model=model_0,
#                data_loader=train_dataloader_simple,
#                loss_fn=nn.CrossEntropyLoss(),
#                optimiser=torch.optim.Adam(params=model_0.parameters(), # Same as usual
#                             lr=0.001) ,
#                accuracy_fn=accuracy_fn,
#                device=device)
    
#     test_step(model=model_0,
#               data_loader=test_dataloader_simple,
#               loss_fn=nn.CrossEntropyLoss(),
#               accuracy_fn=accuracy_fn,
#               device=device)
    
train_time_end_model_0 = timer()

total_train_time_model_0 = print_train_time(start=train_time_start_model_0,
                                            end=train_time_end_model_0,
                                            device=device)

print(f"total train Time:{total_train_time_model_0}")

# Get Model_0 results dictionary

# model_0_results = eval_model(model=model_0,
#                              data_loader=test_dataloader_simple,
#                              loss_fn=nn.CrossEntropyLoss(),
#                              accuracy_fn=accuracy_fn,
#                              device=device)

print(model_0_results)

def plot_loss_curve(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary"""

    # Get the loss values
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="test_loss")
    plt.plot(epochs, test_accuracy, label="test_loss")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

plot_loss_curve(model_0_results)

# Save model_0 state dict
print(f"Saving model 0 to: {MODEL_SAVE_PATH_0}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH_0)
