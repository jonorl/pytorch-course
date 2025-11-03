import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2 Main rules on matrix multiplications

# 1. Inner dimensions must match e.g. (3,2) and (3,2) will error
# but (2,3) and (3,2) or (3,2) and (2,3) should be fine.

# print("error: ", torch.matmul(torch.rand(3,2), torch.rand(3,2)))

# The inner dimensions are the 2nd shape of the first matrix and 1st shape of 2nd matrix
print("fine: ", torch.matmul(torch.rand(2,3), torch.rand(3,2)))

# 2. The resulting matrix has the shape of the outer dimersions (first and last shape)

print("Shape: ", torch.matmul(torch.rand(6,3), torch.rand(3,6)).shape)