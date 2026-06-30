import torch
import torch.nn as nn 

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        # Feature extraction 

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1)
        self.relu1 = nn.ReLU()
        self.pool1  = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)


        # Classification
        self.flattern  = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8,128)
        self.relu3 = nn.ReLU() 
        self.fc2 = nn.Linear(128,10)


    def forward(self,x):
        x = self.pool1(self.relu1(self.conv1(x)))
        print("After Block 1:", x.shape) 
        x = self.pool2(self.relu2(self.conv2(x)))
        print("After Block 2:", x.shape) 
        x = self.flattern(x)
        print("After Flatten:", x.shape) 
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
    

# Test with fake image: batch=1, channels=3 (RGB), 32x32 
model = SimpleCNN()
fake_images  = torch.rand(1,3,32,32)
output = model(fake_images)
print("Final output shape:", output.shape)  # (1, 10)