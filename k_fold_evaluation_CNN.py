import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset, Subset
from trained_ai_2 import Pclass, train_model
import os
from CNN_Image_Scanner_V1 import CNN_Image_Scanner_V1
from torchvision import transforms
import PIL.Image as Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
class Pclass_Categories(Dataset):
    """
    Dataset class for loading image data
    """

    def __init__(self, category, sub_category):
        path = 'C:/Users/yason/PycharmProjects/CNN_Model_Evaluation/expermental_dataset/'
        self.all_images = []
        self.cls_labels = []
        for idx, cls in enumerate(['angry', 'focused', 'happy', 'neutral']):
            class_path = os.path.join(path, category)
            class_path = os.path.join(class_path, sub_category)
            class_path = os.path.join(class_path, cls)
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

def generate_confusion_matrix(test_loader, model):
    """
    Create the confusion matrix of the model by evaluating it in the test set
    :param test_loader dataloader for the test set
    :param model, model to be evaluated
    """
    # Build confusion matrix
    print("Creating confusion matrix...")
    y_pred = []
    y_true = []
    model.eval()
    # iterate over test data
    for inputs, labels in test_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()

        output = model(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).cpu().numpy()  # Moving to CPU for numpy conversion
        y_pred.extend(output)  # Save Prediction

        labels = labels.cpu().numpy()  # Moving to CPU for numpy conversion
        y_true.extend(labels)  # Save Truth

    # constant for classes
    classes = ('angry', 'focused', 'happy', 'neutral')

    # Build confusion matrix
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
def print_metrics(metrics, fold):
    """
    prints metrics for given fold
    :param metrics: dic contains all the metrics of the fold
    :param fold: int number of the fold
    """
    for key, value in metrics.items():
        print(f'Test {key} for fold {fold}: {value}')
        print(f'Test {key} for fold {fold}: {value}')


def print_averages(results_dic, fold_performance_dic):
    """
    prints the averages for the entire fold
    :param results_dic: dic contains all the metrics of each fold
    :param fold_performance_dic: dic contains all the metrics of each fold in numpy value used to acquire mean
    """
    # print results for the all folds
    for key, value in results_dic.items():
        print(f'Results for {key}: {value}')
    # print means for the all folds
    for key, value in fold_performance_dic.items():
        print(f'Average Test {key}: {np.mean(value)}')


def kfold_validation(face_images_set):
    """
    run a kfold validation on a given dataset for a given model
    :param face_images_set: dataset to run the evaluation on
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    results_dic = {}
    fold_performance_dic = {}
    results_loss = {}
    fold_performance_loss = []

    results_accuracy = {}
    fold_performance_accuracy = []

    results_macro_f1 = {}
    fold_performance_macro_f1 = []

    results_micro_f1 = {}
    fold_performance_micro_f1 = []

    results_macro_precision = {}
    fold_performance_macro_precision = []

    results_micro_precision = {}
    fold_performance_micro_precision = []

    results_macro_recall = {}
    fold_performance_macro_recall = []

    results_micro_recall = {}
    fold_performance_micro_recall = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, test_idx) in enumerate(kf.split(face_images_set)):
        train_set = Subset(face_images_set, train_idx)

        test_set = Subset(face_images_set, test_idx)

        train_size = int(0.85 * len(train_set))

        val_size = len(train_set) - train_size
        model = CNN_Image_Scanner_V1()
        model = nn.DataParallel(model)
        model.to(device)
        # Create train and validation datasets
        train_dataset, val_dataset = random_split(train_set, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        metrics = train_model(train_loader, val_loader, test_loader, model, device)
        print_metrics(metrics, fold)
        generate_confusion_matrix(test_loader, model)
        # record metrics
        results_loss[fold] = metrics["loss"]
        fold_performance_loss.append(metrics["loss"])

        results_accuracy[fold] = metrics["accuracy"]
        fold_performance_accuracy.append(metrics["accuracy"])

        results_macro_f1[fold] = metrics["macro_f1"]
        fold_performance_macro_f1.append(metrics["macro_f1"])

        results_micro_f1[fold] = metrics["micro_f1"]
        fold_performance_micro_f1.append(metrics["micro_f1"])

        results_macro_precision[fold] = metrics["macro_precision"]
        fold_performance_macro_precision.append(metrics["macro_precision"])

        results_micro_precision[fold] = metrics["micro_precision"]
        fold_performance_micro_precision.append(metrics["micro_precision"])

        results_macro_recall[fold] = metrics["macro_recall"]
        fold_performance_macro_recall.append(metrics["macro_recall"])

        results_micro_recall[fold] = metrics["micro_recall"]
        fold_performance_micro_recall.append(metrics["micro_recall"])
    # print metrics
    results_dic["loss"] = results_loss
    results_dic["accuracy"] = results_accuracy

    results_dic["macro_precision"] = results_macro_precision
    results_dic["micro_precision"] = results_micro_precision

    results_dic["macro_f1"] = results_macro_f1
    results_dic["micro_f1"] = results_micro_f1

    results_dic["macro_recall"] = results_macro_recall
    results_dic["micro_recall"] = results_micro_recall

    fold_performance_dic["loss"] = fold_performance_loss
    fold_performance_dic["accuracy"] = fold_performance_accuracy

    fold_performance_dic["macro_precision"] = fold_performance_macro_precision
    fold_performance_dic["micro_precision"] = fold_performance_micro_precision

    fold_performance_dic["macro_f1"] = fold_performance_macro_f1
    fold_performance_dic["micro_f1"] = fold_performance_micro_f1

    fold_performance_dic["macro_recall"] = fold_performance_macro_recall
    fold_performance_dic["micro_recall"] = fold_performance_micro_recall

    print_averages(results_dic, fold_performance_dic)


if __name__ == '__main__':

    categories = {"age": ["old", "adult", "young"]}
    print(f"Going over data set:")
    kfold_validation(Pclass('train'))
    """for category in categories.keys():
        print(f"Category: {category}")
        for sub_category in categories[category]:
            images_set = Pclass_Categories(category, sub_category)
            print(f"Sub Category: {sub_category}")
            kfold_validation(images_set)"""
