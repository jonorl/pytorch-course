import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# random_tensor = torch.rand(3,4)
# print(random_tensor)

#Create a random tensor for an image
#Size uses colour channel (RGB), height and width

random_image_size_tensor = torch.rand(size=(3,224,224))
print("shape: ", random_image_size_tensor.shape)
print("dimensions: ", random_image_size_tensor.ndim)

#Here's how to plot this random image
random_image_np = random_image_size_tensor.numpy()

# Plot the image
plt.imshow(random_image_np)
plt.title("Random Image Tensor (H, W, C)")
plt.axis('off') # Hide axis ticks and labels
plt.show()