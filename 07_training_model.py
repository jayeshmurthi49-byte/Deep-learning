import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import pickle 

# Load data
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
train_loader = DataLoader(train_data,batch_size=32,shuffle=True)

# Define MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
        self.relu  = nn.ReLU()


    def forward(self,x):
        x = x.view(-1,28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

# Train
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001) 

for epoch in range(5):
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

# Save model
torch.save(model.state_dict(),'model.pkl')
print("Model saved as model.pkl")