import torch 

x = torch.tensor([0.5, 0.8])
weights = torch.tensor([0.4,0.6])
bias = torch.tensor(0.1)

z = torch.dot(x,weights) + bias
output = torch.sigmoid(z)

print("Weighted sum (z):", z.item())
print("Activation output:", output.item())