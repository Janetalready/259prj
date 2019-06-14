import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, embed_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, (4, embed_dim))
        self.conv2 = nn.Conv2d(1, 2, (3, embed_dim))
        self.conv3 = nn.Conv2d(1, 2, (2, embed_dim))
        self.pooling = nn.AdaptiveMaxPool2d((30,1))
        self.fc1   = nn.Linear(60*3, 9)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out1 = self.pooling(out1)
        out1 = out1.view(out1.shape[0], -1)
        out2 = F.relu(self.conv2(x))
        out2 = self.pooling(out2)
        out2 = out1.view(out2.shape[0], -1)
        out3 = F.relu(self.conv3(x))
        out3 = self.pooling(out3)
        out3 = out3.view(out3.shape[0], -1)
        out = torch.cat([out1,out2,out3], dim=1)
        out = out.view(out.shape[0],-1)
        out = self.fc1(out)
        return out
