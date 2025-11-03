import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = torch.arange(0,100,10, device='cuda', dtype=torch.float32)

print("x: ", x)

# Find min
print(torch.min(x), x.min())

# Find max
print(torch.max(x), x.max())

# Find mean
print(torch.mean(x), x.mean())

# Find the index of the min value
print("argmin / index of min value: ", x.argmin())

# Find the index of the max value
print("argmax / index of max value: ", x.argmax())