import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

random_tensor = torch.rand(3,4)
print(random_tensor)

zeros = torch.zeros(size=(3,4))
print("zeros ",zeros)

print("zeros * random tensor ",zeros*random_tensor)

ones = torch.ones(size=(3,4))
print("ones ", ones)

print("ones.dtype", ones.dtype)