import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjusted for the pooling and conv layers
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)  # Adjusted for the pooling and conv layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def convs(self,x):
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = self.bn2(x)
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = self.bn3(x)
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))
        #print(x.shape)

        if self._to_linear is None:

            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2] # to get the size of 1-d or flattened img
            #print(self._to_linear,x.shape)
        return x
