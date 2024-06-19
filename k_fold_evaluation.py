import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
from trained_ai import Pclass, train_model

from CNN_Image_Scanner_V1 import CNN_Image_Scanner_V1

if __name__ == '__main__':
    face_images_set = Pclass('train')
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
