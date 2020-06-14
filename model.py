import numpy as np
import PIL.Image as Image
from PIL import ImageDraw


import torch.nn as nn

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import glob
import datetime
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CircleNet(nn.Module):    # nn.Module is parent class  
    def __init__(self):
        super(CircleNet, self).__init__()  #calls init of parent class
                
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100,3)
        self.dropout = nn.Dropout(0.25)
          
                    
    def forward(self, x):
        """
        Feed forward through network
        Args:
            x - input to the network
            
        Returns "out", which is the network's output
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc3(x)
        return out
             