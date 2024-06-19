# This file is deprecated and no longer in use.
# Please refer to the newer files for the latest implementation.

import PIL.Image as Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim, nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class Pclass(Dataset):
    def __init__(self, mode):

        path = 'C:/Users/yason/OneDrive/Documents/summer_2024/COMP_472/Part_2_try_1/'
        self.allaimges = []
        self.clsLabel = []
        for idx, cls in enumerate(['angry', 'focused', 'happy', 'neutral']):
            Cpath = os.path.join(path, cls)
            Cpath = os.path.join(Cpath, mode)

            F = os.listdir(Cpath)

            for im in F:
                self.allaimges.append(os.path.join(Cpath, im))
                self.clsLabel.append(idx)
        self.mytransform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                               transforms.ToTensor(),
                                               ])

    def __len__(self):
        return len(self.allaimges)

    def __getitem__(self, idx):

        Im = self.mytransform(Image.open(self.allaimges[idx]))
        Cls = self.clsLabel[idx]

        return Im, Cls


class CNN_Image_Scanner_V1(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.cnn2 = nn.Conv2d(16, 16, 3, padding=1, stride=1)
        self.batch2 = nn.BatchNorm2d(16)

        self.Maxpool1 = nn.MaxPool2d(2)

        self.cnn3 = nn.Conv2d(16, 32, 7, padding=1, stride=1)
        self.batch3 = nn.BatchNorm2d(32)
        self.cnn4 = nn.Conv2d(32, 32, 7, padding=1, stride=1)
        self.batch4 = nn.BatchNorm2d(32)

        self.cnn5 = nn.Conv2d(32, 64, 9, padding=1, stride=1)
        self.batch5 = nn.BatchNorm2d(64)
        self.cnn6 = nn.Conv2d(64, 64, 9, padding=1, stride=1)
        self.batch6 = nn.BatchNorm2d(64)

        self.cnn7 = nn.Conv2d(64, 128, 11, padding=1, stride=1)
        self.batch7 = nn.BatchNorm2d(128)

        self.cnn8 = nn.Conv2d(128, 256, 11, padding=1, stride=1)
        self.batch8 = nn.BatchNorm2d(256)
        self.Maxpool3 = nn.MaxPool2d(4)
        self.fc = nn.Linear(256, 4)

    def forward(self, x):
        x = self.batch1(F.leaky_relu(self.cnn1(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn2(x)))
        x = self.batch2(x)

        x = self.batch3(F.leaky_relu(self.cnn3(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn4(x)))
        x = self.batch4(x)

        x = self.batch5(F.leaky_relu(self.cnn5(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn6(x)))
        x = self.batch6(x)

        x = self.batch7(F.leaky_relu(self.cnn7(x)))
        x = self.Maxpool3(F.leaky_relu(self.cnn8(x)))
        x = self.batch8(x)
        x = self.fc(x.view(x.size(0), -1))
        return x


class CNN_Image_Scanner_V2(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.cnn2 = nn.Conv2d(16, 16, 3, padding=1, stride=1)
        self.batch2 = nn.BatchNorm2d(16)

        self.Maxpool1 = nn.MaxPool2d(2)

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
        self.Maxpool3 = nn.MaxPool2d(4)
        self.fc = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.batch1(F.leaky_relu(self.cnn1(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn2(x)))
        x = self.batch2(x)

        x = self.batch3(F.leaky_relu(self.cnn3(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn4(x)))
        x = self.batch4(x)

        x = self.batch5(F.leaky_relu(self.cnn5(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn6(x)))
        x = self.batch6(x)

        x = self.batch7(F.leaky_relu(self.cnn7(x)))
        x = self.Maxpool3(F.leaky_relu(self.cnn8(x)))
        x = self.batch8(x)
        x = self.fc(x.view(x.size(0), -1))
        return x

class CNN_Image_Scanner_V3(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.cnn2 = nn.Conv2d(16, 16, 3, padding=1, stride=1)
        self.batch2 = nn.BatchNorm2d(16)

        self.Maxpool1 = nn.MaxPool2d(2)

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
        self.cnn8 = nn.Conv2d(128, 256, 9, padding=5, stride=1)
        self.batch8 = nn.BatchNorm2d(256)

        self.cnn9 = nn.Conv2d(256, 512, 11, padding=6, stride=1)
        self.batch9 = nn.BatchNorm2d(512)
        self.cnn10 = nn.Conv2d(512, 512, 11, padding=2, stride=1)
        self.batch10 = nn.BatchNorm2d(512)
        self.Maxpool3 = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, 4)

    def forward(self, x):
        x = self.batch1(F.leaky_relu(self.cnn1(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn2(x)))
        x = self.batch2(x)

        x = self.batch3(F.leaky_relu(self.cnn3(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn4(x)))
        x = self.batch4(x)

        x = self.batch5(F.leaky_relu(self.cnn5(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn6(x)))
        x = self.batch6(x)

        x = self.batch7(F.leaky_relu(self.cnn7(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn8(x)))
        x = self.batch8(x)
        x = self.batch9(F.leaky_relu(self.cnn9(x)))
        x = self.Maxpool3(F.leaky_relu(self.cnn10(x)))
        x = self.batch10(x)
        x = self.fc(x.view(x.size(0), -1))
        return x



class CNN_Image_Scanner_V4(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.Maxpool1 = nn.MaxPool2d(2)

        self.cnn2 = nn.Conv2d(16, 32, 5, padding=1, stride=1)
        self.batch2 = nn.BatchNorm2d(32)

        self.cnn3 = nn.Conv2d(32, 64, 7, padding=1, stride=1)
        self.batch3 = nn.BatchNorm2d(64)

        self.cnn4 = nn.Conv2d(64, 128, 9, padding=1, stride=1)
        self.batch4 = nn.BatchNorm2d(128)

        self.cnn5 = nn.Conv2d(128, 256, 11, padding=1, stride=1)
        self.batch5 = nn.BatchNorm2d(256)
        self.Maxpool3 = nn.MaxPool2d(4)
        self.fc = nn.Linear(25600, 4)

    def forward(self, x):
        x = self.batch1(F.leaky_relu(self.cnn1(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn2(x)))
        x = self.batch2(x)
        x = self.batch3(F.leaky_relu(self.cnn3(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn4(x)))
        x = self.batch4(x)

        x = self.Maxpool3(F.leaky_relu(self.cnn5(x)))
        x = self.batch5(x)

        x = self.fc(x.view(x.size(0), -1))
        return x

class CNN_Image_Scanner_V5(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.cnn2 = nn.Conv2d(16, 16, 3, padding=1, stride=1)
        self.batch2 = nn.BatchNorm2d(16)

        self.Maxpool1 = nn.MaxPool2d(2)

        self.cnn3 = nn.Conv2d(16, 32, 3, padding=1, stride=1)
        self.batch3 = nn.BatchNorm2d(32)
        self.cnn4 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.batch4 = nn.BatchNorm2d(32)

        self.cnn5 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.batch5 = nn.BatchNorm2d(64)
        self.cnn6 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.batch6 = nn.BatchNorm2d(64)

        self.cnn7 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        self.batch7 = nn.BatchNorm2d(128)

        self.cnn8 = nn.Conv2d(128, 256, 3, padding=1, stride=1)
        self.batch8 = nn.BatchNorm2d(256)
        self.Maxpool3 = nn.MaxPool2d(4)
        self.fc = nn.Linear(12544, 4)

    def forward(self, x):
        x = self.batch1(F.leaky_relu(self.cnn1(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn2(x)))
        x = self.batch2(x)

        x = self.batch3(F.leaky_relu(self.cnn3(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn4(x)))
        x = self.batch4(x)

        x = self.batch5(F.leaky_relu(self.cnn5(x)))
        x = self.Maxpool1(F.leaky_relu(self.cnn6(x)))
        x = self.batch6(x)

        x = self.batch7(F.leaky_relu(self.cnn7(x)))
        x = self.Maxpool3(F.leaky_relu(self.cnn8(x)))
        x = self.batch8(x)
        x = self.fc(x.view(x.size(0), -1))
        return x

class CNN_Image_Scanner_V6(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.cnn2 = nn.Conv2d(16, 16, 3, padding=1, stride=1)
        self.batch2 = nn.BatchNorm2d(16)

        self.Maxpool1 = nn.MaxPool2d(2)

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
        self.Maxpool3 = nn.MaxPool2d(4)
        self.fc = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.batch1(F.elu(self.cnn1(x)))
        x = self.Maxpool1(F.elu(self.cnn2(x)))
        x = self.batch2(x)

        x = self.batch3(F.elu(self.cnn3(x)))
        x = self.Maxpool1(F.elu(self.cnn4(x)))
        x = self.batch4(x)

        x = self.batch5(F.elu(self.cnn5(x)))
        x = self.Maxpool1(F.elu(self.cnn6(x)))
        x = self.batch6(x)

        x = self.batch7(F.elu(self.cnn7(x)))
        x = self.Maxpool3(F.elu(self.cnn8(x)))
        x = self.batch8(x)
        x = self.fc(x.view(x.size(0), -1))
        return x

class CNN_Image_Scanner_V7(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.cnn2 = nn.Conv2d(16, 16, 3, padding=1, stride=1)
        self.batch2 = nn.BatchNorm2d(16)

        self.Maxpool1 = nn.MaxPool2d(2)

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
        self.Maxpool3 = nn.MaxPool2d(4)
        self.fc = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.batch1(F.relu(self.cnn1(x)))
        x = self.Maxpool1(F.relu(self.cnn2(x)))
        x = self.batch2(x)

        x = self.batch3(F.relu(self.cnn3(x)))
        x = self.Maxpool1(F.relu(self.cnn4(x)))
        x = self.batch4(x)

        x = self.batch5(F.relu(self.cnn5(x)))
        x = self.Maxpool1(F.relu(self.cnn6(x)))
        x = self.batch6(x)

        x = self.batch7(F.relu(self.cnn7(x)))
        x = self.Maxpool3(F.relu(self.cnn8(x)))
        x = self.batch8(x)
        x = self.fc(x.view(x.size(0), -1))
        return x
class CNN_Image_Scanner_V7(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.cnn2 = nn.Conv2d(16, 16, 3, padding=1, stride=1)
        self.batch2 = nn.BatchNorm2d(16)

        self.Maxpool1 = nn.MaxPool2d(2)

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
        self.Maxpool3 = nn.MaxPool2d(4)
        self.fc = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.batch1(F.relu(self.cnn1(x)))
        x = self.Maxpool1(F.relu(self.cnn2(x)))
        x = self.batch2(x)

        x = self.batch3(F.relu(self.cnn3(x)))
        x = self.Maxpool1(F.relu(self.cnn4(x)))
        x = self.batch4(x)

        x = self.batch5(F.relu(self.cnn5(x)))
        x = self.Maxpool1(F.relu(self.cnn6(x)))
        x = self.batch6(x)

        x = self.batch7(F.relu(self.cnn7(x)))
        x = self.Maxpool3(F.relu(self.cnn8(x)))
        x = self.batch8(x)
        x = self.fc(x.view(x.size(0), -1))
        return x

class CNN_Image_Scanner_V8(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = nn.Conv2d(1, 16, 3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.cnn2 = nn.Conv2d(16, 16, 3, padding=1, stride=1)
        self.batch2 = nn.BatchNorm2d(16)

        self.Maxpool1 = nn.MaxPool2d(2)

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
        self.Maxpool3 = nn.MaxPool2d(4)
        self.fc = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.batch1(F.relu(self.cnn1(x)))
        x = self.Maxpool1(F.relu(self.cnn2(x)))
        x = self.batch2(x)

        x = self.batch3(F.elu(self.cnn3(x)))
        x = self.Maxpool1(F.elu(self.cnn4(x)))
        x = self.batch4(x)

        x = self.batch5(F.elu(self.cnn5(x)))
        x = self.Maxpool1(F.elu(self.cnn6(x)))
        x = self.batch6(x)

        x = self.batch7(F.relu(self.cnn7(x)))
        x = self.Maxpool3(F.relu(self.cnn8(x)))
        x = self.batch8(x)
        x = self.fc(x.view(x.size(0), -1))
        return x

def evaluate_metrics(preds, labels, num_classes):
    # Flatten tensors
    preds = preds.view(-1)
    labels = labels.view(-1)
    print("all_preds shape", all_preds.shape)
    print("all_labels shape", all_labels.shape)
    # Initialize confusion matrix
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    # Populate confusion matrix
    for t, p in zip(labels, preds):
        conf_matrix[t.long(), p.long()] += 1
    print(conf_matrix)
    # Calculate metrics
    true_positives = torch.diag(conf_matrix)
    false_positives = conf_matrix.sum(dim=0) - true_positives
    false_negatives = conf_matrix.sum(dim=1) - true_positives
    print("true_positives", true_positives)
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)
    accuracy = true_positives.sum().item() / conf_matrix.sum().item()

    return accuracy, precision.mean().item(), recall.mean().item(), f1.mean().item()


if __name__ == '__main__':

    batch_size = 64
    test_batch_size = 64
    input_size = 3 * 32 * 32  # 3 channels, 32x32 image size
    hidden_size = 50  # Number of hidden units
    output_size = 10  # Number of output classes (CIFAR-10 has 10 classes)
    num_epochs = 10
    patience = 5

    train_set = Pclass('train')
    test_set = Pclass('test')
    val_set = Pclass('val')
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)
    epochs = 15
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN_Image_Scanner_V8()
    model = nn.DataParallel(model)
    model.to(device)
    # model.load_state_dict(torch.load('path'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    BestACC = 0
    best_val_loss = -1
    for epoch in range(epochs):
        print(f"Epoch:{epoch}")
        model.train()
        epochs_no_improve = 0
        running_loss = 0

        rightPred: float = 0
        totalPred: float = 1
        for images, labels in train_loader:
            # print(f"insantce: {images} label:{labels}")
            optimizer.zero_grad()
            output = model(images.cuda())
            loss = criterion(output, labels.cuda())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.cuda().size(0)
            predictedClass, preds = torch.max(output, 1)
            rightPred += (preds == labels.cuda()).sum().item()
            totalPred += labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = rightPred / totalPred
        print(f"train_accuracy:{train_accuracy}")
        print(f"train_loss:{train_loss}")

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            allsamps = 0
            rightPred = 0

            for images, labels in val_loader:
                output = model(images.cuda())
                val_loss = criterion(output, labels.cuda())
                val_running_loss += val_loss.item() * images.cuda().size(0)
                predictedClass, preds = torch.max(output, 1)
                val_correct += (preds == labels.cuda()).sum().item()
                val_total += labels.size(0)

            ACC = val_correct / val_total
            print('val Accuracy is=', ACC * 100)
            if ACC > BestACC:
                BestACC = ACC
                # torch.save(model.state_dict())
                # torch.save(model.state_dict(), 'path')
        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = val_correct / val_total
        print(f"Val_loss:{val_loss}")
        print(f"Val_accuracy:{val_accuracy}")
        if val_loss < best_val_loss or best_val_loss == -1:
            best_val_loss = val_loss
            print("best_val_loss:", best_val_loss)
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            print("No improvement")
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print('Early stopping!')
            break
    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        allsamps = 0
        rightPred = 0

        for images, labels in test_loader:
            output = model(images.cuda())
            test_loss = criterion(output, labels.cuda())
            test_running_loss += test_loss.item() * images.cuda().size(0)
            predictedClass, preds = torch.max(output, 1)
            test_correct += (preds == labels.cuda()).sum().item()
            test_total += labels.size(0)
            all_preds.append(preds.cuda())
            all_labels.append(labels.cuda())

        ACC = test_correct / test_total
        print('val Accuracy is=', ACC * 100)
        if ACC > BestACC:
            BestACC = ACC
            # torch.save(model.state_dict())
            # torch.save(model.state_dict(), 'path')
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    test_loss = test_running_loss / len(test_loader.dataset)
    test_accuracy = test_correct / test_total
    evaluate_metrics(all_preds, all_labels, 4)
