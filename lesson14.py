import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# How to create a range

print(torch.arange(start=0,end=1000, step=77))

# Creating tensors like
one_to_ten = torch.arange(1, 11)
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)