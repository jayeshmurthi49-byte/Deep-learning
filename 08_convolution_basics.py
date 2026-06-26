import torch 
import torch.nn as nn 

# Step 1: Create a fake grayscale image (1 image, 1 channel, 6x6)
image = torch.tensor([[
    [1., 0., 1., 0., 1., 0.],
    [0., 1., 0., 1., 0., 1.],
    [1., 0., 1., 0., 1., 0.],
    [0., 1., 0., 1., 0., 1.],
    [1., 0., 1., 0., 1., 0.],
    [0., 1., 0., 1., 0., 1.]
]]).unsqueeze(0) 

print("Image shape:",image.shape)

# Step 2: Apply convolution - 1 filter, 3x3 kernel, no padding 
conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=0)
feature_map = conv(image)
print("Feature map shape (no padding):", feature_map.shape) 

# Step 3: Apply convolution - 3 filters 
conv3 = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=1,padding=0)
output3 = conv3(image)
print("With 3 filters:", output3.shape) 

# Step 4: Apply convolution - stride 2 
conv_s2 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=2,padding=0)
output_s2 = conv_s2(image)
print("With stride 2:", output_s2.shape)  

# Step 5: Apply convolution - with padding 
conv_pad = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
output_pad = conv_pad(image)
print("With padding=1:", output_pad.shape)