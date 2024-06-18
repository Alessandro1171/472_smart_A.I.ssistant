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
import CNN_Image_Scanner_V1
import CNN_Image_Scanner_V2
import CNN_Image_Scanner_V3

class Pclass(Dataset):
    """
    Gets the separate datasets for the different faces and have them corresponds to their appropriate labels
    """
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

def train_model(train_set_loader: DataLoader, val_set_loader: DataLoader, test_set_loader: DataLoader, model):
    """
    trains the given model on a train test and then evaluates the performance for a number of epochs
    the best version of the model is saved and early stooping is implemented if accuracy doesn't improve over a certain amount of epochs
    :param train_set_loader: dataloader for the train set
    :param val_set_loader: dataloader for the validation set
    :param test_set_loader: dataloader for the test set
    :param model: model to be trained
    """
    epochs = 15
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    best_acc = 0
    best_val_loss = -1
    patience = 5
    for epoch in range(epochs):
        print(f"Epoch:{epoch}")
        model.train()
        epochs_no_improve = 0
        running_loss = 0

        rightPred: float = 0
        totalPred: float = 1
        # train the model
        for images, labels in train_set_loader:
            optimizer.zero_grad()
            output = model(images.cuda())
            loss = criterion(output, labels.cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.cuda().size(0)
            predictedClass, preds = torch.max(output, 1)
            rightPred += (preds == labels.cuda()).sum().item()
            totalPred += labels.size(0)
        train_loss = running_loss / len(train_set_loader.dataset)
        train_accuracy = rightPred / totalPred
        print(f"train_accuracy:{train_accuracy}")
        print(f"train_loss:{train_loss}")

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        # evaluate the model
        with torch.no_grad():
            for images, labels in val_set_loader:
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

        val_loss = val_running_loss / len(val_set_loader.dataset)
        val_accuracy = val_correct / val_total
        print(f"Val_loss:{val_loss}")
        print(f"Val_accuracy:{val_accuracy}")
        # saves best model based on loss
        if val_loss < best_val_loss or best_val_loss == -1:
            best_val_loss = val_loss
            print("best_val_loss:", best_val_loss)
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            print("No improvement")
            epochs_no_improve += 1
        # early stopping if no improvement is found
        if epochs_no_improve >= patience:
            print('Early stopping!')
            break
    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss()
    # final evaluation on the test set
    with torch.no_grad():
        for images, labels in test_set_loader:
            output = model(images.cuda())
            test_loss = criterion(output, labels.cuda())
            test_running_loss += test_loss.item() * images.cuda().size(0)
            predictedClass, preds = torch.max(output, 1)
            test_correct += (preds == labels.cuda()).sum().item()
            test_total += labels.size(0)

        ACC = test_correct / test_total
        print('Test Accuracy is=', ACC * 100)


if __name__ == '__main__':
    # get the train, test and validation sets
    train_set = Pclass('train')
    test_set = Pclass('test')
    val_set = Pclass('val')
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)
    # create model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN_Image_Scanner_V2()
    model = nn.DataParallel(model)
    model.to(device)
    train_model(train_loader, val_loader, test_loader, model)


