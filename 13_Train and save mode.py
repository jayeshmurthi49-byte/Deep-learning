import torch
import torch.nn as nn
from torchvision import models,transforms,datasets
from torch.utils.data import DataLoader
from PIL import Image 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder('/content/cats-vs-dogs/PetImages',transform=transform)
print("Classes:", dataset.classes)
print("Total images:", len(dataset))

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data,test_data  = torch.utils.data.random_split(dataset,[train_size,test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False 

model.fc = nn.Linear(512,2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device) 
print("Using device:", device) 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(),lr=0.001) 

for epoch in range(3):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss   = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images,labels in test_loader:
        images,labels = images.to(device),labels.to(device)
        output = model(images)
        _,predicted = torch.max(output,1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
print(f"Test Accuracy: {100 * correct / total:.2f}%") 

torch.save(model.state_dict(), 'cats_vs_dogs_model.pkl')
print("Model saved!")


