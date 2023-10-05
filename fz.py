import torch
from torch import nn
import numpy as np

seq_length = 5
batch_size = 1
input_size = 100

data = torch.randn(seq_length,batch_size,input_size)
layer = nn.LSTM(input_size=100, hidden_size=100, num_layers=10)
outputs, (h_n, c_n) = layer(data)
print(outputs, outputs.shape)