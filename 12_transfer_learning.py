import torch
import torch.nn as nn 
from torchvision import models 


model = models.resnet18(pretrained=True)
print("Original ResNet18 final layer:", model.fc) 

for param in model.parameters():
    param.requires_grad  = False


model.fc = nn.Linear(512,2)
print("Modified final layer:", model.fc)

trainable = [name for name,param in model.named_parameters() if param.requires_grad]
print("Trainable layers:", trainable) 

fake_image = torch.rand(1, 3, 224, 224)  # ResNet expects 224x224
output = model(fake_image)
print("Output shape:", output.shape) 

for param in model.layer4.parameters():
    param.requires_grad = True

trainable_finetune = [name for name,param in model.named_parameters() if param.requires_grad]
print("Trainable layers after fine tuning:", trainable_finetune)