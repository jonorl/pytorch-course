import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tensor_a = torch.tensor([[1,2],
                        [3,4],
                        [5,6]],dtype=torch.float32, device='cuda')

tensor_b = torch.tensor([[7,10],
                        [8,11],
                        [9,12]],dtype=torch.float32, device='cuda')

#print("matmul: ", torch.mm(tensor_a, tensor_b))  torch.mm is the same as torch.matmul

# To fix out tensor shape issues, we can manipulate the shape of our tensors using transpose
matMulResult = torch.mm(tensor_a, tensor_b.T)
print("matmul with transposed tensor B (.T): ", matMulResult)
print("matMulResult shape: ", matMulResult.shape)