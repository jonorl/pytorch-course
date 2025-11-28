## Make and evaluate random predictions with best model

import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path 
from torchmetrics import Accuracy, ConfusionMatrix
from timeit import default_timer as timer
from tqdm.auto import tqdm
import pandas as pd
import random
from mlxtend.plotting import plot_confusion_matrix

### SETUP

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True) # Create models directory if it doesn't exist

MODEL_NAME_2 = "fashion_mnist_model_2.pth"
MODEL_SAVE_PATH_2 = MODEL_PATH / MODEL_NAME_2

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


def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            
            # Prepare the sample (add a batch dimension and pass to target device)
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # Forward pass (model outputs raw logits)
            pred_logit = model(sample)

            # Get pred prob (logit -> pred prob)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0
                                      )
            
            # Get pred_prob off the GPU for further calculation
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)


## Models

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

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape, IMAGE_SIZE, IMAGE_SIZE)
            x = self.conv_block_1(dummy_input)
            x = self.conv_block_2(x)
            flattened_size = x.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_size, # multiply by results of conv blocks 1 & 2
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
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
IMAGE_SIZE = image.shape[1]

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

## Loading models

# Initiate model instances
loaded_model_2 = FashionMNISTModelV2(input_shape=1,
                                      hidden_units=10,
                                      output_shape=len(class_names))

# Load state dict for model_2
print(f"Loading model 1 state dict from: {MODEL_SAVE_PATH_2}")
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH_2))
loaded_model_2.to(device)

# Model 2 - CNN

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Measure time
train_time_start_model_2 = timer()

# Create optimisation and eval loop using train_step() and test_step()

for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}\n-----------")
    train_step(model=loaded_model_2,
               data_loader=train_dataloader,
               loss_fn=nn.CrossEntropyLoss(),
               optimiser=torch.optim.SGD(params=loaded_model_2.parameters(), # Same as usual
                            lr=0.1) ,
               accuracy_fn=accuracy_fn,
               device=device)
    
    test_step(model=loaded_model_2,
              data_loader=test_dataloader,
              loss_fn=nn.CrossEntropyLoss(),
              accuracy_fn=accuracy_fn,
              device=device)
    
train_time_end_model_2 = timer()

total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                            end=train_time_end_model_2,
                                            device=device)
print("total train Time:", total_train_time_model_2)

# Get Model_2 results dictionary

model_2_results = eval_model(model=loaded_model_2,
                             data_loader=test_dataloader,
                             loss_fn=nn.CrossEntropyLoss(),
                             accuracy_fn=accuracy_fn,
                             device=device)

# random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9): # k is how many random samples we have
    test_samples.append(sample)
    test_labels.append(label)

# View the first sample shape
test_samples[0].shape

# plt.imshow(test_samples[0].squeeze(), cmap='gray')
# plt.title(class_names[test_labels[0]])
# plt.show()

# Make predictions
pred_probs = make_predictions(model=loaded_model_2,
                              data=test_samples)

print(pred_probs[:2])

# Convert prediction probs into labels
pred_classes = pred_probs.argmax(dim=1)
print(pred_classes)

# Plot predictions

plt.figure(figsize=(9,9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    # Create subplot
    plt.subplot(nrows, ncols, i+1)

    # Plot the target image
    plt.imshow(sample.squeeze(), cmap='gray')

    # Find the prediction (in text form e.g. sandal)
    pred_label = class_names[pred_classes[i]]

    # Get the truth label
    truth_label = class_names[test_labels[i]]

    # Create a title for the plot
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c='g')
    else:
        plt.title(title_text, fontsize=10, c='r')
    plt.axis(False)
    plt.show()

## Making a confusion matrix for further evaluation

y_preds = []
loaded_model_2.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions..."):
        # Send the data and targets to device
        X, y = X.to(device), y.to(device)
        # Do the forward pass
        y_logit = loaded_model_2(X)
        # Turn predictions from logits -> pred probs
        y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
        # Put predictions on CPU for evaluation
        y_preds.append(y_pred.cpu())

# Concatenate list of predictions into a tensor

# print(y_preds)
y_pred_tensor = torch.cat(y_preds)
print(y_pred_tensor[:10])

# Set up confusion instance and compare prediction to target

confmat = ConfusionMatrix(task="multiclass", num_classes=NUM_CLASSES)
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets) # targets = labels

# Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), #matplotlib likes working with numpy
    class_names=class_names,
    figsize=(10,7)
)

