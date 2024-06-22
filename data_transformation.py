import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import PIL.Image as Image
import os


class GaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)  # Clamp to maintain pixel range


class Augmented_Dataset(Dataset):
    def __init__(self, transformer):
        self.transform = transformer
        path = "C:/Users/aless/Documents/472/472_smart_A.I.ssistant/image_archive/"
        self.image_files = []
        self.clsLabel = []
        for idx, cls in enumerate(['angry', 'focused', 'happy', 'neutral']):
            Cpath = os.path.join(path, cls)

            F = os.listdir(Cpath)

            for im in F:
                self.image_files.append(os.path.join(Cpath, im))
                self.clsLabel.append(idx)
        self.mytransform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                               transforms.ToTensor(),
                                               ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        Im = self.mytransform(Image.open(self.image_files[idx]))
        Cls = self.clsLabel[idx]
        #if self.transform:
        #    Im = self.transform(Im)
        return Im, Cls


class Self_Storing_Augmented_Dataset(Dataset):
    def __init__(self, transformer, store_dir, path, name_extenstion, label):
        self.transform = transformer
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        self.image_names = []
        self.clsLabel = []
        self.image_files = []
        self.ext = name_extenstion
        for idx, cls in enumerate(['angry', 'focused', 'happy', 'neutral']):
            if label == cls:
                Cpath = os.path.join(path, cls)

                F = os.listdir(Cpath)
                y = self.image_names + [f for f in os.listdir(Cpath) if f.endswith('.png')]
                self.image_names = y
                for im in F:
                    self.image_files.append(os.path.join(Cpath, im))
                    self.clsLabel.append(idx)
        print(f"self.image_files:{self.image_files}")
        self.mytransform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                               transforms.ToTensor(),
                                               ])
        print(f"self.image_files length:{len(self.image_files)}")
        print(f"self.clsLabel length:{len(self.clsLabel)}")
        print(f"self.image_names length:{len(self.image_names)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        Im = self.mytransform(Image.open(self.image_files[idx]))
        Cls = self.clsLabel[idx]
        if self.transform:
            Im = self.transform(Im)
        print(f"What is iniiiiiiiiiiiiiiiiiiiii here:{self.image_names[idx]}")
        file_name = self.ext + self.image_names[idx]
        save_path = os.path.join(self.store_dir, file_name)
        print(f"save_path:{save_path}")
        image_pil = transforms.ToPILImage()(Im)
        image_pil.save(save_path)

        return Im, Cls


transform1 = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(128),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    GaussianNoise(mean=0.0, std=0.05),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])
transform2 = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(25, 25)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])
transform3 = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(128),
    transforms.GaussianBlur(kernel_size=11, sigma=(0.3, 3.0)),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])
transform4 = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.3, hue=0.15),
    transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
    GaussianNoise(mean=0.0, std=0.075),
])
transform5 = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.5, hue=0.2),
    transforms.GaussianBlur(kernel_size=11, sigma=(0.3, 1.0)),
    GaussianNoise(mean=0.0, std=0.1),
])
transform6 = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(128),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.4, 2.5)),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])
transform7 = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(128),
    GaussianNoise(mean=0.0, std=0.25),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])
transform8 = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(128),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    GaussianNoise(mean=0.0, std=0.05),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])

experiment = Augmented_Dataset(transform1)
angry_old_transformers = [transform1,transform2,transform3,transform4,transform5,transform6,transform7]
angry_old_transformers_name = ["transform1","transform2","transform3","transform4","transform5","transform6","transform7"]

happy_old_transformers = [transform1,transform2,transform3,transform4,transform5,transform6,transform7]
happy_old_transformers_name = ["transform1","transform2","transform3","transform4","transform5","transform6","transform7"]
train_loader = DataLoader(experiment, batch_size=16, shuffle=True, num_workers=0, drop_last=True)

print(f"train_loader:{len(train_loader)}")
for images, labels in train_loader:
    print(images.size(), labels)
for i in range(0, len(angry_old_transformers)):

    experiment1 = Self_Storing_Augmented_Dataset(angry_old_transformers[i], "C:/Users/aless/Documents/472/saved/old/angry/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/age/old/", angry_old_transformers_name[i], "angry")
    train_loader1 = DataLoader(experiment1, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        print("Hello!")
