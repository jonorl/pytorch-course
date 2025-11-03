import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Create a tensor
tensor = torch.tensor([1,2,3],dtype=torch.float32, device='cuda')

# Add 10 to it
print("tensor + 10: ", tensor + 10)

# Multiply by 10 
print("tensor * 10: ", tensor * 10)

# Subtract 10
print("tensor - 10: ", tensor - 10)

# Alternatively pytorch in built method:
print("torch.mul(tensor,10) ", torch.mul(tensor,10))

# Matrix multiplication
print("tensor.dtype: ", tensor.dtype)
print("torch.matmul(tensor, tensor)", torch.matmul(tensor, tensor))
# In this case it'll be 1*1 + 2*2 + 3*3 = 14

# Matrix multiplication using a loop

start_time = time.time()
value = 0
for i in range(len(tensor)):
    value += tensor[i] * tensor[i]
print(value)
torch.cuda.synchronize()
end_time = time.time()
print("Value:", value)
print(f"Time taken: **{end_time - start_time:.6f} seconds**") #Time taken: **0.000788 seconds**


start_time = time.time()

print("torch.matmul(tensor, tensor)", torch.matmul(tensor, tensor))
torch.cuda.synchronize()
end_time = time.time()
print("Value:", value)
print(f"Time taken: **{end_time - start_time:.6f} seconds**") #Time taken: **0.000598 seconds**

