"""
Trains a PyTorch image classification model using device-agnostic code
"""

from pathlib import Path
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils
from timeit import default_timer as timer
from torchmetrics import Accuracy

NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE_WIDTH = 64
IMAGE_SIZE_LENGTH = 64
INPUT_SHAPE = 3
MODEL_NAME="modular_tinyvgg.pth"

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
OPTIMISER = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

# Start the timer
start_time = timer()

# Start training with help from enginge.py
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