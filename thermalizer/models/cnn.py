import torch.nn as nn
import numpy as np

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv1d(1,64,kernel_size=3,padding='same', padding_mode='circular')
        self.act1=nn.ReLU()
        self.conv2=nn.Conv1d(64,64,kernel_size=3,padding='same', padding_mode='circular')
        self.act2=nn.ReLU()
        self.conv3=nn.Conv1d(64,1,kernel_size=3,padding='same', padding_mode='circular')

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x
