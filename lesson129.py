from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

def walk_through_dir(dir_path):
    """Walks through dir_path returning its own content"""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

print(walk_through_dir(image_path))

# random.seed(42)

image_path_list = list(image_path.glob("*/*/*.jpg"))

random_image_path = random.choice(image_path_list)

print(random_image_path)

image_class = random_image_path.parent.stem

img = Image.open(random_image_path)
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"image height {img.height}")
print(f"image width: {img.width}")
# Image._show(img)

# Turn an image into an array
img_as_array = np.asarray(img)

# Plot it
# plt.figure(figsize=(10,7))
# plt.imshow(img_as_array)
# plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
# plt.axis(False)
# plt.show()

data_transform = transforms.Compose([
    # Resize to 64x64
    transforms.Resize(size=(64,64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5),
    # Turn image to tensor
    transforms.ToTensor()
])

print(f"image path: {image_path}")

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """
    Selects random images from a path of images and loads/transforms
    them then plots the original vs the transofrmed version
    """

    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)

            # Transofmr and plot target image
            transformed_image = transform(f).permute(1,2,0) #permute swaps the order of the axis (color channel 3 last)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

# plot_transformed_images(image_paths=image_path_list, 
#                         transform=data_transform,
#                         n=3,
#                         seed=42)

# Transform folders into tensor datasets

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform, # A transform for the data
                                  target_transform=None) # transform for the target/label

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

class_names = train_data.classes

print(f"test_data:{test_data}")
print(f"train_data:{train_data}")
print(f"class names: {train_data.classes}")
class_dict = train_data.class_to_idx
print(f"class dict: {class_dict}")
print(train_data.samples[0])

img, label = train_data[0][0], train_data[0][1]
print(f"Image tensor:\n {img}")
print(f"Image shape:\n {img.shape}")
print(f"Image datatype:\n {img.dtype}")
print(f"Image label:\n {label}")
print(f"Image label name:\n {class_names[label]}")
print(f"Image label datatype:\n {type(label)}")

# Rearrange the order dimensions (color at the end)
img_permute = img.permute(1, 2, 0)

# Print different shapes
print(f"Original shape:{img.shape} -> color, height, width")
print(f"Image permute: {img_permute.shape} -> height, width, color")

# Plot the image
# plt.figure(figsize=(10, 7))
# plt.imshow(img_permute)
# plt.axis("off")
# plt.title(class_names[label], fontsize=14)
# plt.show()

from torch.utils.data import DataLoader
BATCH_SIZE = 32
CPU_COUNT = os.cpu_count()
print(f"Number of CPU cores: {CPU_COUNT}")

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=CPU_COUNT,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              num_workers=CPU_COUNT,
                              shuffle=False)

img, label = next(iter(train_dataloader))
