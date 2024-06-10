import torch
from torch import optim, nn
from trained_ai import CNN_Image_Scanner_V1, Pclass
from torch.utils.data import DataLoader
import glob
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
import PIL.Image as Image
class Get_Image(Dataset):
    def __init__(self, path, label):
        self.allaimges = []
        self.clsLabel = []

        self.allaimges.append(path)
        self.clsLabel.append(label)
        self.mytransform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                            transforms.ToTensor(),
                                           ])


    def __len__(self):
        return len(self.allaimges)

    def __getitem__(self, idx):


        Im=self.mytransform(Image.open(self.allaimges[idx]))
        Cls=self.clsLabel[idx]

        return Im,


model = CNN_Image_Scanner_V1()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()  # Set to evaluation mode
user_input = input("do yo want to be measured on a single image(A) or the test set(B):A or B")
testset = None
if user_input == "A":
    path_input = input("specify the path:")
    label = input("specify the image label:")
    testset = Get_Image(path_input, label)
else:
    testset = Pclass('test')

test_loader = DataLoader(testset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
BestACC=0
with torch.no_grad():
    allsamps = 0
    rightPred = 0

    for instances, labels in test_loader:
            output = model(instances.detach().clone())
            predictedClass, preds = torch.max(output, 1)
            allsamps += output.size(0)
            rightPred += (torch.max(output, 1)[1] == labels.detach().clone()).sum()
            ACC = rightPred / allsamps

    ACC = rightPred / allsamps
    print('Accuracy is=', ACC * 100)
    if ACC > BestACC:
        BestACC = ACC
    print("Best Accuracy :", BestACC)

