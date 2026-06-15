import torch 

import torch

x = torch.tensor([0.5, 0.8])
weights = torch.tensor([0.4, 0.6], requires_grad=True)
bias = torch.tensor(0.1, requires_grad=True)

z = torch.dot(x, weights) + bias
output = torch.sigmoid(z) 

target = torch.tensor(1.0)
loss = torch.nn.functional.binary_cross_entropy(output,target)
loss.backward() 

print("Output:", output.item())
print("Loss:", loss.item()) 
print("Gradient w.r.t weights:", weights.grad)
print("Gradient w.r.t bias:", bias.grad)