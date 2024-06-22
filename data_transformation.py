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
    transforms.RandomCrop(size=(118, 118)),
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
    transforms.RandomResizedCrop(168),
    transforms.GaussianBlur(kernel_size=13, sigma=(0.25, 2.0)),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])
transform9 = transforms.Compose([
    transforms.GaussianBlur(kernel_size=13, sigma=(0.25, 1.0)),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])
transform10 = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(168),
    transforms.ColorJitter(brightness=0.35, contrast=0.2, saturation=0.55, hue=0.05),
])
experiment = Augmented_Dataset(transform1)
angry_old_transformers = [transform1,transform2,transform3,transform4,transform5,transform6,transform7, transform8]
angry_old_transformers_name = ["transform1","transform2","transform3","transform4","transform5","transform6","transform7", "transform8"]

happy_old_transformers = [transform1,transform2,transform3,transform4,transform5,transform6,transform7]
happy_old_transformers_name = ["transform1","transform2","transform3","transform4","transform5","transform6","transform7"]

neutral_old_transformers = [transform1,transform2,transform3,transform4,transform5,transform6,transform7]
neutral_old_transformers_name = ["transform1","transform2","transform3","transform4","transform5","transform6","transform7"]

focused_old_transformers = [transform1,transform2,transform3,transform4,transform5,transform6,transform7,transform8,transform9,transform10]
focused_old_transformers_name = ["transform1","transform2","transform3","transform4","transform5","transform6","transform7", "transform8", "transform9","transform10"]
train_loader = DataLoader(experiment, batch_size=16, shuffle=True, num_workers=0, drop_last=True)

print(f"train_loader:{len(train_loader)}")
for images, labels in train_loader:
    print(images.size(), labels)
"""
for i in range(0, len(angry_old_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(angry_old_transformers[i], "C:/Users/aless/Documents/472/saved/old/angry/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/age/old/", angry_old_transformers_name[i], "angry")
    train_loader1 = DataLoader(experiment1, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        print("Hello!")

for i in range(0, len(happy_old_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(happy_old_transformers[i], "C:/Users/aless/Documents/472/saved/old/happy/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/age/old/", happy_old_transformers_name[i], "happy")
    train_loader1 = DataLoader(experiment1, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        print("Hello!")

for i in range(0, len(neutral_old_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(neutral_old_transformers[i], "C:/Users/aless/Documents/472/saved/old/neutral/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/age/old/", neutral_old_transformers_name[i], "neutral")
    train_loader1 = DataLoader(experiment1, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        print("Hello!")

for i in range(0, len(focused_old_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(focused_old_transformers[i], "C:/Users/aless/Documents/472/saved/old/focused/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/age/old/", focused_old_transformers_name[i], "focused")
    train_loader1 = DataLoader(experiment1, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        print("Hello!")


angry_young_transformers = [transform1,transform2,transform3,transform4]
angry_young_transformers_name = ["transform1","transform2","transform3","transform4"]

happy_young_transformers = [transform1,transform3,transform5,transform7]
happy_young_transformers_name = ["transform1","transform3","transform5","transform7"]

neutral_young_transformers = [transform2,transform4,transform6,transform8]
neutral_young_transformers_name = ["transform2","transform4","transform6","transform8"]

focused_young_transformers = [transform7,transform6,transform5,transform3]
focused_young_transformers_name = ["transform7","transform6","transform5","transform3"]

for i in range(0, len(angry_young_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(angry_young_transformers[i], "C:/Users/aless/Documents/472/saved/young/angry/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/age/young/", angry_young_transformers_name[i], "angry")
    train_loader1 = DataLoader(experiment1, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        print("Hello!")

for i in range(0, len(happy_young_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(happy_young_transformers[i], "C:/Users/aless/Documents/472/saved/young/happy/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/age/young/", happy_young_transformers_name[i], "happy")
    train_loader1 = DataLoader(experiment1, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        print("Hello!")

for i in range(0, len(neutral_young_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(neutral_young_transformers[i], "C:/Users/aless/Documents/472/saved/young/neutral/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/age/young/", neutral_young_transformers_name[i], "neutral")
    train_loader1 = DataLoader(experiment1, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        print("Hello!")

for i in range(0, len(focused_young_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(focused_young_transformers[i], "C:/Users/aless/Documents/472/saved/young/focused/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/age/young/", focused_young_transformers_name[i], "focused")
    train_loader1 = DataLoader(experiment1, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        print("Hello!")



angry_women_transformers = [transform1, transform4, transform5, transform8]
angry_women_transformers_name = ["transform1", "transform4", "transform5", "transform8"]
counter = 0
for i in range(0, len(angry_women_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(angry_women_transformers[i], "C:/Users/aless/Documents/472/saved/women/angry/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/gender/women/", angry_women_transformers_name[i], "angry")
    train_loader1 = DataLoader(experiment1, batch_size=25, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        counter = counter+25
        print("Hello!")
        if counter >= 700:
            break
    if counter >= 700:
        break
happy_women_transformers = [transform1,transform3,transform5,transform7]
happy_women_transformers_name = ["transform1","transform3","transform5","transform7"]
counter = 0
for i in range(0, len(happy_women_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(happy_women_transformers[i], "C:/Users/aless/Documents/472/saved/women/happy/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/gender/women/", happy_women_transformers_name[i], "happy")
    train_loader1 = DataLoader(experiment1, batch_size=25, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        counter = counter+25
        print("Hello!")
        if counter >= 400:
            break
    if counter >= 400:
        break
counter=0
neutral_women_transformers = [transform2,transform4,transform6,transform8]
neutral_women_transformers_name = ["transform2","transform4","transform6","transform8"]
for i in range(0, len(neutral_women_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(neutral_women_transformers[i], "C:/Users/aless/Documents/472/saved/women/neutral/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/gender/women/", neutral_women_transformers_name[i], "neutral")
    train_loader1 = DataLoader(experiment1, batch_size=25, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        counter = counter+25
        print("Hello!")
        if counter >= 400:
            break
    if counter >= 400:
        break
focused_women_transformers = [transform7,transform6,transform5,transform3]
focused_women_transformers_name = ["transform7","transform6","transform5","transform3"]

counter = 0
for i in range(0, len(focused_women_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(focused_women_transformers[i], "C:/Users/aless/Documents/472/saved/women/focused/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/gender/women/", focused_women_transformers_name[i], "focused")
    train_loader1 = DataLoader(experiment1, batch_size=25, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        counter = counter+25
        print("Hello!")
        if counter >= 400:
            break
    if counter >= 400:
        break
"""

angry_men_transformers = [transform1, transform4, transform5, transform8]
angry_men_transformers_name = ["transform1", "transform4", "transform5", "transform8"]
counter = 0
for i in range(0, len(angry_men_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(angry_men_transformers[i], "C:/Users/aless/Documents/472/saved/men/angry/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/gender/men/", angry_men_transformers_name[i], "angry")
    train_loader1 = DataLoader(experiment1, batch_size=25, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        counter = counter+25
        print("Hello!")
        if counter >= 400:
            break
    if counter >= 400:
        break

happy_men_transformers = [transform1,transform3,transform5,transform7]
happy_men_transformers_name = ["transform1","transform3","transform5","transform7"]
counter = 0
for i in range(0, len(happy_men_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(happy_men_transformers[i], "C:/Users/aless/Documents/472/saved/men/happy/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/gender/men/", happy_men_transformers_name[i], "happy")
    train_loader1 = DataLoader(experiment1, batch_size=25, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        counter = counter+25
        print("Hello!")
        if counter >= 700:
            break
    if counter >= 700:
        break

counter=0
neutral_men_transformers = [transform2,transform4,transform6,transform8]
neutral_men_transformers_name = ["transform2","transform4","transform6","transform8"]
for i in range(0, len(neutral_men_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(neutral_men_transformers[i], "C:/Users/aless/Documents/472/saved/men/neutral/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/gender/men/", neutral_men_transformers_name[i], "neutral")
    train_loader1 = DataLoader(experiment1, batch_size=25, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        counter = counter+25
        print("Hello!")
        if counter >= 200:
            break
    if counter >= 200:
        break

focused_men_transformers = [transform1,transform6,transform5,transform3]
focused_men_transformers_name = ["transform1","transform6","transform5","transform3"]

counter = 0
for i in range(0, len(focused_men_transformers)):
    experiment1 = Self_Storing_Augmented_Dataset(focused_men_transformers[i], "C:/Users/aless/Documents/472/saved/men/focused/",
                                                 "C:/Users""/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/gender/men/", focused_men_transformers_name[i], "focused")
    train_loader1 = DataLoader(experiment1, batch_size=25, shuffle=True, num_workers=0, drop_last=True)
    for images, labels in train_loader1:
        counter = counter+25
        print("Hello!")
        if counter >= 300:
            break
    if counter >= 300:
        break