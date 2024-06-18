import torch
from torch import optim, nn
from trained_ai import CNN_Image_Scanner_V1, Pclass
from torch.utils.data import DataLoader
import glob
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from CNN_Image_Scanner_V1 import CNN_Image_Scanner_V1
from CNN_Image_Scanner_V2 import CNN_Image_Scanner_V2
from CNN_Image_Scanner_V3 import CNN_Image_Scanner_V3
import seaborn as sns


class Get_Image(Dataset):
    """
    Custom Dataset for loading images
    """
    def __init__(self, path, label):
        self.allaimges = []
        self.clsLabel = []
        if label == "angry":
            self.clsLabel.append(0)
        elif label == "focused":
            self.clsLabel.append(1)
        elif label == "happy":
            self.clsLabel.append(2)
        else:
            self.clsLabel.append(3)
        self.allaimges.append(path)
        self.mytransform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                               transforms.ToTensor(),
                                               ])

    def __len__(self):
        return len(self.allaimges)

    def __getitem__(self, idx):
        Im = self.mytransform(Image.open(self.allaimges[idx]))
        Cls = self.clsLabel[idx]

        return Im, Cls


def emotion_translator(result):
    """
    converts the class from an int to a string
    :param result class int
    :return class string
    """
    emotion_number = result.item()
    if result == 0:
        return "angry"
    elif result == 1:
        return "focused"
    elif result == 2:
        return "happy"
    else:
        return "neutral"


def generate_confusion_matrix(testloader, model):
    """
    Create the confusion matrix of the model by evaluating it in the test set
    :param testloader dataloader for the test set
    :param model model to be evaluated
    """
    # Build confusion matrix
    print("Creating confusion matrix...")
    y_pred = []
    y_true = []
    model.eval()
    # iterate over test data
    for inputs, labels in testloader:
        inputs = inputs.cuda()
        labels = labels.cuda()

        output = model(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).cpu().numpy()  # Moving to CPU for numpy conversion
        y_pred.extend(output)  # Save Prediction

        labels = labels.cpu().numpy()  # Moving to CPU for numpy conversion
        y_true.extend(labels)  # Save Truth

    # constant for classes
    classes = ('angry', 'focused', 'happy', 'neutral')

    # Building confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix / np.sum(cf_matrix, axis=1)[:, None])
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt='f', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('output.png')
    plt.show()


def single_image_guess(model):
    """
    a model is evaluated on a single image and its prediction is given
    :model, model to be evaluated
    """
    path_input = input("specify the path:")
    label = input("specify the image label:")
    test_set = Get_Image(path_input, label)
    print(f"test_set:{len(test_set)}")
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0, drop_last=False)
    rightPred = 0
    totalPred = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()

            output = model(images)
            predictedClass, preds = torch.max(output, 1)
            rightPred += (preds == labels).sum().item()
            print(f"Predicted Class:{preds.cpu().numpy()},  Actual Class:{labels.cpu().numpy()}")  # Move to CPU for printing

        if rightPred > 0:
            print("Model guessed right!")

        else:
            print("Model guessed wrong!")
        pred_class = emotion_translator(preds)
        act_class = emotion_translator(labels)
        totalPred += labels.size(0)
        print(f"What the AI predicted:{pred_class},  the class actually was:{act_class}")


def test_set_evaluation(model):
    """
    evaluates a models performance on a test set and then displays a confusion matrix and the precision,
    recall and f1 both micro and marco
    :param testloader dataloader for the test set
    :param model model to be evaluated
    """
    test_set = Pclass('test')
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    generate_confusion_matrix(test_loader, model)
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss()
    BestACC = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        allsamps = 0
        rightPred = 0
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()

            output = model(images)
            test_loss = criterion(output, labels)
            test_running_loss += test_loss.item() * images.size(0)
            predictedClass, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())  # Move to CPU for numpy conversion
            all_labels.extend(labels.cpu().numpy())  # Move to CPU for numpy conversion
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

        ACC = test_correct / test_total
        print('Test Accuracy is=', ACC * 100)
        if ACC > BestACC:
            BestACC = ACC
    display_values(all_preds, all_labels, BestACC)


def display_values(all_preds, all_labels, BestACC):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    micro_precision = precision_score(all_labels, all_preds, average='micro')
    micro_recall = recall_score(all_labels, all_preds, average='micro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')

    print("Macro Precision: {:.4f}".format(macro_precision))
    print("Macro Recall: {:.4f}".format(macro_recall))
    print("Macro F1: {:.4f}".format(macro_f1))
    print("Micro Precision: {:.4f}".format(micro_precision))
    print("Micro Recall: {:.4f}".format(micro_recall))
    print("Micro F1: {:.4f}".format(micro_f1))
    print(f"BestACC:{BestACC}")
    report = classification_report(all_labels, all_preds, target_names=['angry', 'focused', 'happy', 'neutral'])
    print("Classification Report:")
    print(report)


model = CNN_Image_Scanner_V1().cuda()  # Just to ensure that model is on GPU
model = nn.DataParallel(model)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()  # Setting to evaluation mode
user_input = input("do you want to be measured on a single image(A) or the test set(B) chose A or B:")
testset = None
if user_input == "A":
    single_image_guess(model)
else:
    test_set_evaluation(model)
