import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a tensor, 1 2D matrix with 3 arrays containing 3 ints

x = torch.arange(1,10).reshape(1,3,3)
print("x: ", x) 
print("x shape: ", x.shape)

# Let's index on the first (and only) table
print("x[0]: ", x[0])

# Let's index on the first row/dim;
print("x[0][0]: ", x[0][0]) #also x[0,0] is fine

# Let's index on the first item of the first row/dim;
print("x[0][0]: ", x[0][0][0])

# You can also use ":" to select "all" of target dimension
print("x[:,:,0]: ", x[:,:,0]) # first col