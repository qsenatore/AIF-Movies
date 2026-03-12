import torch
import torch.nn as nn
import torch.nn.functional as F

class MovieNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.bn   = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1     = nn.Linear(128 * 23 * 37, 512)
        self.drop1   = nn.Dropout(p=0.3)
        
        self.fc2     = nn.Linear(512, 256)
        self.drop2   = nn.Dropout(p=0.1)
        
        self.fc3     = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.bn(self.conv3(x))))
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        
        x = self.fc3(x)
        return x


