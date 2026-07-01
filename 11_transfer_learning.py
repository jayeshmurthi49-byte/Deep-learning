import torch
import torch.nn as nn
from torchvision import models

# Step 1: Load pretrained ResNet18
model = models.resnet18(pretrained=True)
print("Original ResNet18 final layer:", model.fc)

# Step 2: Freeze all layers (feature extraction mode)
for param in model.parameters():
    param.requires_grad = False

# Step 3: Replace final layer for binary classification (cats vs dogs = 2 classes)
model.fc = nn.Linear(512, 2)
print("Modified final layer:", model.fc)

# Step 4: Check which layers are trainable
trainable = [name for name, param in model.named_parameters() if param.requires_grad]
print("Trainable layers:", trainable)

# Step 5: Test forward pass with fake image
fake_image = torch.rand(1, 3, 224, 224)  # ResNet expects 224x224
output = model(fake_image)
print("Output shape:", output.shape)  # (1, 2)

# Step 6: Fine tuning - unfreeze last layer block
for param in model.layer4.parameters():
    param.requires_grad = True

trainable_finetune = [name for name, param in model.named_parameters() if param.requires_grad]
print("Trainable layers after fine tuning:", trainable_finetune)