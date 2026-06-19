import torch
import torch.nn as nn
from torchvision import datasets,transforms 
from torch.utils.data import DataLoader

# Step 1: Load MNIST data 
transform = transforms.ToTensor() 

train_data = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_data = datasets.MNIST(root='./data',train=False,download=True,transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data,batch_size=32,shuffle=False)

# Step 2: Define MLP 
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(28*28,128) # input: 784 pixels 
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
        self.relu = nn.ReLU() 

    def forward(self,X):
        X = X.view(-1,28*28)    # flatten image
        X = self.relu(self.fc1(X))
        X = self.relu(self.fc2(X))
        X = self.fc3(X)
        return X 
    
#  Step 3: Model, loss, optimizer
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001) 

# Step 4: Training loop 
for epoch in range(3):
    model.train()
    total_loss = 0
    for images,labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Step 5: Test accuracy 
model.eval()
correct = 0
total = 0 

with torch.no_grad():
    for images,labels in test_loader:
        output = model(images)
        _,predicted = torch.max(output,1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
print(f"Test Accuracy: {100 * correct / total:.2f}%")