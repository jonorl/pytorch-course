"""
Trains a PyTorch image classification model using device-agnostic code
"""

import argparse
from pathlib import Path
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils
from timeit import default_timer as timer
from torchmetrics import Accuracy

# Create argument parser
parser = argparse.ArgumentParser(description="Train a PyTorch image classification model")

# Add arguments
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train for")
parser.add_argument("--batch_size", type=int, default=32, help="Number of samples per batch")
parser.add_argument("--hidden_units", type=int, default=10, help="Number of hidden units in model")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
parser.add_argument("--image_size_width", type=int, default=64, help="Image width")
parser.add_argument("--image_size_length", type=int, default=64, help="Image height")
parser.add_argument("--model_name", type=str, default="modular_tinyvgg.pth", help="Name for saved model")

# Parse arguments
args = parser.parse_args()

# Assign parsed arguments to variables
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
IMAGE_SIZE_WIDTH = args.image_size_width
IMAGE_SIZE_LENGTH = args.image_size_length
MODEL_NAME = args.model_name

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_SHAPE = 3

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

data_transform = transforms.Compose([
    transforms.Resize(size=(IMAGE_SIZE_WIDTH, IMAGE_SIZE_LENGTH)),
    transforms.ToTensor()
])

INPUT_SIZE = data_transform.transforms[0].size[0]

# Create DataLoaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=data_transform,
                                                                               batch_size=BATCH_SIZE)

# Create model
model = model_builder.TinyVGG(input_shape=INPUT_SHAPE,
                                hidden_units=HIDDEN_UNITS,
                                output_shape=len(class_names),
                                input_size=INPUT_SIZE).to(DEVICE)

# Setup loss and optimiser
LOSS_FN = torch.nn.CrossEntropyLoss()
OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Start the timer
start_time = timer()

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=LOSS_FN,
             optimizer=OPTIMISER,
             epochs=NUM_EPOCHS,
             device=DEVICE,
             accuracy_fn=Accuracy)

# End timer
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model to file
utils.save_model(model=model,
                 target_dir="models",
                 model_name=MODEL_NAME)