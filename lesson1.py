import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# print(torch.__version__)

#Scalar - which is kinda like an int
scalar = torch.tensor(7)
print(scalar)

print("scalar dimensions: ",scalar.ndim)
print("scalar shape: ", scalar.shape)

#Vector - like an array
vector = torch.tensor([7,7])
print(vector)
print("vector dimensions: ", vector.ndim)
print("vector shape: ", vector.shape)

#MATRIX - 2D array...?
MATRIX = torch.tensor([[7,7], 
                      [9,10]])
print(MATRIX)
print("matrix dimensions: ", MATRIX.ndim)
print("matrix shape: ", MATRIX.shape)

#TENSOR - 3D array...? Or many 2D arrays
TENSOR = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
print(TENSOR)
print("tensor dimensions: ", TENSOR.ndim)
print("tensor shape: ", TENSOR.shape)