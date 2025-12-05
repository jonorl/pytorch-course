
from pathlib import Path
import torch
from torchvision import transforms
from going_modular import data_setup, model_builder, engine

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

data_transform = transforms.Compose([
    # Resize to 64x64
    transforms.Resize(size=(64,64)),
    # Turn image to tensor
    transforms.ToTensor()
])

INPUT_SIZE = data_transform.transforms[0].size[0]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=data_transform,
                                                                               batch_size=32)

print(train_dataloader)
print(test_dataloader)
print(class_names)
print(INPUT_SIZE)

torch.manual_seed(42)
model_1 = model_builder.TinyVGG(input_shape=3,
                                hidden_units=10,
                                output_shape=len(class_names),
                                input_size=INPUT_SIZE).to(DEVICE)

print(f"Model 1: {model_1}")

