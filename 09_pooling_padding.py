import torch 
import torch.nn as nn

image = torch.rand(1,1,8,8)
print("Orginal image shape",image.shape) 

# Step 1: Max Pooling 2x2 
max_pool = nn.MaxPool2d(kernel_size=2,stride=2)
output_max = max_pool(image)
print("After MaxPool2d:", output_max.shape)  # (1, 1, 4, 4) 

# Step 2: Average Pooling 2x2 
avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)
output_avg = avg_pool(image)
print("After AvgPool2d:", output_avg.shape)  # (1, 1, 4, 4) 

# Step 3: Conv with no padding 
conv_no_pad = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
out_no_pad = conv_no_pad(image)
print("Conv no padding:", out_no_pad.shape)  # (1, 1, 6, 6) 

# Step 4: Conv with padding=1 (same size output)
conv_pad = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
out_pad = conv_pad(image)
print("Conv with padding=1:",out_pad.shape)  # (1, 1, 8, 8) same as input 

# Step 5: Conv with stride=2 
conv_stride = nn.Conv2d(1,1,kernel_size=3,stride=2,padding=0)
out_stride = conv_stride(image)
print("Conv with stride=2:", out_stride.shape)  # (1, 1, 3, 3) 

# Step 6: Conv + MaxPool combined (like in real CNN) 
conv = nn.Conv2d(1,8,kernel_size=3,stride=1,padding=1)
pool = nn.MaxPool2d(2,2)
x = conv(image)
print("After Conv:", x.shape)    # (1, 8, 8, 8)
x = pool(x)
print("After Pool:", x.shape) 
