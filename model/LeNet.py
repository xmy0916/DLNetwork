import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import save_feature


class LeNet(nn.Module):
    def __init__(self,saveFeature = False,cfg = None):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.saveFeature = saveFeature
        self.cfg = cfg

    def forward(self, x):
        x = self.conv1(x)
        if self.saveFeature:
            save_feature(x, "lenet", "conv1")
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        if save_feature:
            save_feature(x, "lenet", "conv2")
        x = self.relu(x)
        x = self.maxpool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    net = LeNet()
    print(net)
