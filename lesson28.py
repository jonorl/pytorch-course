import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Numpy to pytorch

array = np.arange (1.0, 8.0)
tensor = torch.from_numpy(array)

print("tensor", tensor)
print("dtype", array.dtype) # be aware that numpys default is float64 instead of torch's float32

## Pytorch to numpy

tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print("tensor", tensor)
print("numpy_tensor", numpy_tensor)
print("numpy_tensor dtype", numpy_tensor.dtype)

#they pass by value, so changing one won't change the other

tensor = tensor + 1

print("tensor", tensor)
print("numpy_tensor", numpy_tensor)
