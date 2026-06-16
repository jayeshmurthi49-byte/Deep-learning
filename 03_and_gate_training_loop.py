import torch

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([0.,0.,0.,1.]) 

weights = torch.zeros(2,requires_grad=True)
bias = torch.zeros(1,requires_grad=True)

lr = 0.1 

for epochs in range(20):
    total_loss = 0
    for i in range(len(X)):
        z = torch.dot(X[i],weights) + bias
        output =  torch.sigmoid(z)
        loss = torch.nn.functional.binary_cross_entropy(output,y[i].unsqueeze(0))
        loss.backward()

        with torch.no_grad():
            weights -= lr * weights.grad
            bias -= lr*bias.grad
            weights.grad.zero_()
            bias.grad.zero_()

        total_loss += loss.item()
        print(f"Epoch {epochs+1} , Loss {total_loss:4f}")

print("Final weights:", weights)
print("Final bias:", bias)