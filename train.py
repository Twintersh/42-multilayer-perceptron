import pandas as pd
from typing import Tuple
import numpy as np
import os

def getData(foldername: str) -> Tuple[np.array, np.array, np.array, np.array]:
    train_folder = os.path.join(foldername, "train.csv")
    predict_folder = os.path.join(foldername, "predict.csv")

    dataset_train = pd.read_csv(train_folder, header=None)
    dataset_predict = pd.read_csv(predict_folder, header=None)

    train_data = dataset_train.to_numpy()[:,2:]
    train_label = dataset_train.to_numpy()[:,1]
    pred_data = dataset_predict.to_numpy()[:,2:]
    pred_label = dataset_predict.to_numpy()[:,1]
    return (train_data, train_label, pred_data, pred_label)


if __name__ == "__main__":
    train_data, train_label, pred_data, pred_label = getData("datasets")
        
    dataset_train = pd.read_csv("datasets/train.csv", header=None)
    dataset_predict = pd.read_csv("datasets/predict.csv", header=None)

    train_data = dataset_train.to_numpy()[:,2:]
    train_label = dataset_train.to_numpy()[:,1]
    pred_data = dataset_predict.to_numpy()[:,2:]
    pred_label = dataset_predict.to_numpy()[:,1]

