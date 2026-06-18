import torch
import torch.nn as nn  


# Step 1: Define the MLP using nn.Module

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP,self).__init__()
        self.layer1 = nn.Linear(2,4)  # 2 inputs -> 4 hidden neurons
        self.layer2 = nn.Linear(4,1)  # 4 hidden -> 1 output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() 

    def forward(self,X):
        X = self.relu(self.layer1(X))  # hidden layer + ReLU
        X = self.sigmoid(self.layer2(X))  # output layer + Sigmoid 
        return X
    
# Step 2: Create model, loss function, optimizer 
model = SimpleMLP()
criterion = nn.BCELoss() 
optimizer = torch.optim.Adam(model.parameters(),lr=0.01) 

# Step 3: Dummy data (AND gate) 
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[0.],[0.],[1.]]) 

# Step 4: Training loop 
for epoch in range(20):
    optimizer.zero_grad() # clear old gradients
    output = model(X) # forward pass
    loss = criterion(output,y) # compute loss
    loss.backward() # backward pass
    optimizer.step()   # update weights

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}") 


# Step 5: Check predictions 
print("\nFinal Predictions:")
with torch.no_grad():
    pred = model(X)
    print(pred)