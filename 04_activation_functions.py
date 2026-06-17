import torch

X = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])


# Sigmoid
sigmoid = torch.sigmoid(X)
print("Sigmoid:", sigmoid) 


# Tanh
tanh = torch.tanh(X)
print("Tanh:", tanh)


# ReLU
relu = torch.relu(X)
print("ReLU:", relu)

# Softmax (needs 1D input, dim=0) 

softmax = torch.softmax(X,dim=0)
print("Softmax:",softmax.sum().item())


