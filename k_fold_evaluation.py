import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
from trained_ai import Pclass, train_model
import os
from CNN_Image_Scanner_V1 import CNN_Image_Scanner_V1
from torchvision import transforms
import PIL.Image as Image


class Pclass_Categoires(Dataset):
    """
    Gets the separate datasets for the different faces and have them corresponds to their appropriate labels
    """

    def __init__(self, category, sub_category):

        path = 'C:/Users/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/'
        self.allaimges = []
        self.clsLabel = []
        for idx, cls in enumerate(['angry', 'focused', 'happy', 'neutral']):
            Cpath = os.path.join(path, category)
            Cpath = os.path.join(Cpath, sub_category)
            Cpath = os.path.join(Cpath, cls)
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


def kfold_validation(data_set_category, data_set_sub_category):
    face_images_set = Pclass_Categoires(data_set_category, data_set_sub_category)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    results_loss = {}
    fold_performance_loss = []
    results_accuracy = {}
    fold_performance_accuracy = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(face_images_set)):
        train_set = Subset(face_images_set, train_idx)
        test_set = Subset(face_images_set, test_idx)
        train_size = int(0.85 * len(train_set))
        val_size = len(train_set) - train_size
        model = CNN_Image_Scanner_V1()
        # Create train and validation datasets
        print("train_size:", type(train_set))
        print(f"train_size:{train_set}")
        train_dataset, val_dataset = random_split(train_set, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        accuracy, avg_test_loss = train_model(train_loader, val_loader, test_loader, model)

        results_loss[fold] = avg_test_loss
        fold_performance_loss.append(avg_test_loss)

        results_accuracy[fold] = accuracy
        fold_performance_accuracy.append(accuracy)
        print(f'Test Loss for fold {fold}: {avg_test_loss}')
        print(f'Test Accuracy for fold {fold}: {accuracy}')

    print(f'Results: {results_accuracy}')
    print(f'Average Validation Loss: {np.mean(fold_performance_accuracy)}')

    print(f'Results: {results_loss}')
    print(f'Average Validation Loss: {np.mean(fold_performance_loss)}')


if __name__ == '__main__':
    categories = {"age": ["old", "adult", "young"], "gender": ["men", "women"]}
    for category in categories.keys():
        print(f"Going over category:{category}")
        for sub_category in categories[category]:
            kfold_validation(category, sub_category)
