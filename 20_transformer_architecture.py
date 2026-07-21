import torch
import torch.nn as nn
import math

# Step 1: Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Step 2: Single Attention Head
def attention(Q, K, V):
    dk = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)

# Step 3: Multi Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.size()
        Q = self.W_Q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        attn_output = attention(Q, K, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_O(attn_output)

# Step 4: Test it
d_model = 16
num_heads = 2
seq_len = 5
batch_size = 1

x = torch.rand(batch_size, seq_len, d_model)
print("Input shape:", x.shape)

pe = PositionalEncoding(d_model)
x_pe = pe(x)
print("After Positional Encoding:", x_pe.shape)

mha = MultiHeadAttention(d_model, num_heads)
output = mha(x_pe)
print("After Multi Head Attention:", output.shape)