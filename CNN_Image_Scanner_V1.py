import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim, nn


class CNN_Image_Scanner_V1(nn.Module):
    """AI that trains itself on facial images and their labeled classes to be able to predict the expression of an
    individual and whether they are feeling happy, angry, focused or neutral"""

    def __init__(self):
        super().__init__()

        self.cnn1 = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.cnn2 = nn.Conv2d(16, 16, 3, padding=1, stride=1)
        self.batch2 = nn.BatchNorm2d(16)

        self.max_pool1 = nn.MaxPool2d(2)

        self.cnn3 = nn.Conv2d(16, 32, 5, padding=1, stride=1)
        self.batch3 = nn.BatchNorm2d(32)
        self.cnn4 = nn.Conv2d(32, 32, 5, padding=1, stride=1)
        self.batch4 = nn.BatchNorm2d(32)

        self.cnn5 = nn.Conv2d(32, 64, 7, padding=1, stride=1)
        self.batch5 = nn.BatchNorm2d(64)
        self.cnn6 = nn.Conv2d(64, 64, 7, padding=1, stride=1)
        self.batch6 = nn.BatchNorm2d(64)

        self.cnn7 = nn.Conv2d(64, 128, 9, padding=1, stride=1)
        self.batch7 = nn.BatchNorm2d(128)

        self.cnn8 = nn.Conv2d(128, 256, 9, padding=1, stride=1)
        self.batch8 = nn.BatchNorm2d(256)
        self.max_pool2 = nn.MaxPool2d(4)
        self.fc = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.batch1(F.relu(self.cnn1(x)))
        x = self.max_pool1(F.relu(self.cnn2(x)))
        x = self.batch2(x)

        x = self.batch3(F.elu(self.cnn3(x)))
        x = self.max_pool1(F.elu(self.cnn4(x)))
        x = self.batch4(x)

        x = self.batch5(F.elu(self.cnn5(x)))
        x = self.max_pool1(F.elu(self.cnn6(x)))
        x = self.batch6(x)

        x = self.batch7(F.relu(self.cnn7(x)))
        x = self.max_pool2(F.relu(self.cnn8(x)))
        x = self.batch8(x)
        x = self.fc(x.view(x.size(0), -1))
        return x
