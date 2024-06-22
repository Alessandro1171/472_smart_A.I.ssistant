import PIL.Image as Image
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from CNN_Image_Scanner_V1 import CNN_Image_Scanner_V1  # Replace with your model import
from CNN_Image_Scanner_V2 import CNN_Image_Scanner_V2  # Replace with your model import
from CNN_Image_Scanner_V3 import CNN_Image_Scanner_V3  # Replace with your model import
from sklearn.metrics import precision_score, recall_score, f1_score

class Pclass(Dataset):
    """
    Dataset class for loading image data
    """

    def __init__(self, mode):
        path = 'C:/Users/aless/Documents/472/472_smart_A.I.ssistant/expermental_dataset/'
        self.all_images = []
        self.cls_labels = []
        for idx, cls in enumerate(['angry', 'focused', 'happy', 'neutral']):
            class_path = os.path.join(path, cls)
            class_path = os.path.join(class_path, mode)
            file_list = os.listdir(class_path)

            for im in file_list:
                self.all_images.append(os.path.join(class_path, im))
                self.cls_labels.append(idx)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.cls_labels[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return image, label


def train_model(train_loader, val_loader, test_loader, model, device):
    """
    Function to train and evaluate the model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 15
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Train Loss: {train_loss}")

        model.eval()
        val_running_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                val_loss = criterion(output, labels)
                val_running_loss += val_loss.item()
                _, preds = torch.max(output, 1)
                val_correct += (preds == labels).sum().item()

            val_loss = val_running_loss / len(val_loader)
            val_accuracy = val_correct / len(val_loader.dataset)
            print(f"Validation Loss: {val_loss}, Accuracy: {val_accuracy}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), 'best_model.pth')
                print("Saved best model")

            else:
                epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print('Early stopping!')
            break

    # Testing the model
    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            test_loss = criterion(output, labels)
            test_running_loss += test_loss.item()
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())  # Move to CPU for numpy conversion
            all_labels.extend(labels.cpu().numpy())
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    test_loss = test_running_loss / len(test_loader)
    test_accuracy = test_correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss}, Accuracy: {test_accuracy}")
    print(f"all preds size:{len(all_preds)} all all_labels size:{len(all_labels)}")
    print(f"all preds {all_preds}")
    print(f"all labels {all_labels}")

    mi_pr = precision_score(all_labels, all_preds, average='micro', zero_division=0)

    mi_rc = recall_score(all_labels, all_preds, average='micro', zero_division=0)

    mi_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    mac_pr = precision_score(all_labels, all_preds, average='macro', zero_division=0)

    mac_rc = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    mac_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    metrics = {"macro_precision": mac_pr,
               "macro_recall": mac_rc,
               "macro_f1": mac_f1,
               "micro_precision": mi_pr,
               "micro_recall": mi_rc,
               "micro_f1": mi_f1,
               "loss": test_loss,
               "accuracy": test_accuracy}

    return metrics

