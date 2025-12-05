### Contains functions for training and testing a PyTorch model

from typing import Dict, List, Tuple
import torch 
from torch import nn
from tqdm.auto import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
               device=DEVICE):
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
               device: torch.device = DEVICE):
    
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
              device: torch.device = DEVICE):
    
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
          device=DEVICE
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
