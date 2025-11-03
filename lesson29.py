import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_device('cuda') # leave this one to default device to CUDA

## Reproducbility (trying to take random out of random)

# To do this you add a *seed* to make it a pseudo randomness

RANDOM_SEED = 42 # Any number will do really
random_tensor_A = torch.rand(3,4)
random_tensor_B = torch.rand(3,4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B) #false

torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3,4)
torch.manual_seed(RANDOM_SEED) # You need to manually set it up each time
random_tensor_D = torch.rand(3,4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D) #true