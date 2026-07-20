import torch
import torch.nn.functional as F
import math 


# Step 1: Create fake Q, K, V vectors
# Imagine 4 words, each represented as 8 dimensional vector 

seq_len = 4
d_model = 8

Q = torch.rand(seq_len,d_model) 
K = torch.rand(seq_len,d_model)
V = torch.rand(seq_len,d_model)

# Step 2: Compute attention scores 
scores = torch.matmul(Q,K.transpose(-2,-1) / math.sqrt(d_model))
print("\nAttention scores shape:", scores.shape)
print("Attention scores:\n", scores) 

# Step 3: Apply softmax to get attention weights 
weights = F.softmax(scores,dim=1)
print("\nAttention weights (after softmax):\n", weights.round(decimals=2))
print("Each row sums to ",weights.sum(dim=1))  


# Step 4: Multiply weights by V to get output
output = torch.matmul(weights,V)
print("\n Attention output shape:",output.shape)
print("\n Attention output :",output) 
