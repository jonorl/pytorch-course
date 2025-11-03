import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Reshaping, stacknig, squeezing and unsqueezing tensors

# Reshaping - reshapes an  input tensor to a defined shape
# View - return a view on an input tensor of a certain shape but keep the same memory as og tensor
# Stacking - Combine multiple tensors on top of eachother(vstack) or side by side (hstack)
# Squeeze - removes all '1' dimesions from tensor
# Unsqueeze - add a '1' dimension to a target tensor
# Permute -  Return a view of the input with dimensions permuted (swapped) in a certain way

x = torch.arange(0.,9.,device='cuda',dtype=torch.float32)
print("x: ", x)
print("x shape: ", x.shape)

# We're reshaping a 1D tensor into a 2D table with 3 rows and 3 cols instead:
x_reshaped = x.reshape(3,3)
print("x reshaped: ", x_reshaped)
print("x reshaped shape: ", x_reshaped.shape)

# Change the view
z = x.view(3,3)
print("z: ", z)
print("z shape: ", z.shape)

# This is kinda like pass-by-reference where the new alias changes the og
z[:,0] = 5
print("x: ", x, "z: ", z)

# Stacking tensors

x_stacked = torch.stack([x,x,x,x], dim=0) #dim 0 is horizontal and dim 1 is vertical
print("x stacked: ",x_stacked)


## Squeeze: Removes 1 dimension:

# Let's reshape x back to a 2D matrix of 1 row:
x_reshaped = x.reshape(1,9)
print("x_reshaped", x_reshaped)
print("x_reshaped.shape", x_reshaped.shape)

# squeeze will convert the 2D matrix of 1 row to a single row/array/1D
x_squeezed = x_reshaped.squeeze()
print("x_reshaped.squeeze()", x_squeezed)
print("x_reshaped.squeeze().shape", x_squeezed.shape)

## Unsqueeze add a single dimension to a target tensor at a specific dim

x_unsqueezed = x_squeezed.unsqueeze(dim=0) #dim 0 adds a dimension as a row, dim1 as col
print("x_unsqueezed", x_unsqueezed)
print("x_unsqueezed.squeeze().shape", x_unsqueezed.shape)

## Torch permute: Change the order of the dimensions

x_original = torch.rand(size=(224,224,3)) # random image

# Permute the og tensor to rearrange the axis/dim order

x_permuted = x_original.permute(2,0,1) # shifts axis 0->1, 1->2, 2->0

print("x_original.shape", x_original.shape)

print("x_permuted.shape", x_permuted.shape) # colour channels, height, width

# Permuting is also  like pass-by-reference where the new alias changes the og
x_original[0,0,0] = 728218
print(x_permuted[0,0,0])

