import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tensor data types

# Float32 tensor

float_32_tensor = torch.tensor([3.0, 6.0, 9.0], 
                               dtype=None, # Data types, like float16, 32, etc 
                               device='cuda', # CPU or GPU
                               requires_grad=False) # wheter or not to track gradients with this tensors operations

print(float_32_tensor)
print("data type: ", float_32_tensor.dtype)
print("Device: ", float_32_tensor.device) # this will show cuda:0


# This is to check what cuda:0 is, which in my case is AMD Radeon RX 6950 XT
device_index = 0
gpu_name = torch.cuda.get_device_name(device_index)
print(f"Device Name at {float_32_tensor.device}: **{gpu_name}**")

# How to convert a float32 tensor to float16

float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)

print(float_16_tensor * float_32_tensor)

### 3 Main errors from working with tensors:

# 1. Wrong datatype - check with tensor.dtype
# 2. Wrong shape - check with tensor.shape
# 3. Tensors not on the right device - check with tensor.device 

some_tensor = torch.rand(3,4,device='cuda')
print("some rand tensor: ", some_tensor)
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}") # or .size()
print(f"Device of tensor: {some_tensor.device}")