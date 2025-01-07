import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import matplotlib.pyplot as plt

def setLabelsValues(label_array: np.array) -> np.array:
	label_mapping = {'M': 1, 'B': 0}
	try:
		return np.array([label_mapping[label] for label in label_array])
	except KeyError:
		exit(1)

def normalizeData(data: np.array) -> np.array:
	data_scaler = StandardScaler()
	return data_scaler.fit_transform(data)

def getDataFromDataset(foldername: str) -> Tuple[np.array, np.array, np.array, np.array]:
	try:
		# Construct file paths
		train_folder = os.path.join(foldername, "train.csv")
		predict_folder = os.path.join(foldername, "predict.csv")

		# Load datasets
		try:
			dataset_train = pd.read_csv(train_folder, header=None)
		except FileNotFoundError:
			raise FileNotFoundError(f"Train file not found: {train_folder}")
		except pd.errors.EmptyDataError:
			raise ValueError(f"Train file is empty or corrupted: {train_folder}")

		try:
			dataset_predict = pd.read_csv(predict_folder, header=None)
		except FileNotFoundError:
			raise FileNotFoundError(f"Predict file not found: {predict_folder}")
		except pd.errors.EmptyDataError:
			raise ValueError(f"Predict file is empty or corrupted: {predict_folder}")

		# Extract and process labels
		tmp_train_label = dataset_train.to_numpy()[:, 1]
		tmp_pred_label = dataset_predict.to_numpy()[:, 1]
		train_label = setLabelsValues(tmp_train_label)
		pred_label = setLabelsValues(tmp_pred_label)

		# Extract and normalize data
		tmp_train_data = dataset_train.to_numpy()[:, 2:]
		tmp_pred_data = dataset_predict.to_numpy()[:, 2:]
		train_data = normalizeData(tmp_train_data)
		pred_data = normalizeData(tmp_pred_data)

		return (train_data, train_label, pred_data, pred_label)

	except Exception as e:
		print(f"An error occurred while processing the dataset: {e}")
		exit(1)

def plotGraphs(epochs_history, accuracy_train, accuracy_pred, loss_history, loss_pred_history):
		fig, axes = plt.subplots(1, 2, figsize=(12, 6))
		axes[0].plot(epochs_history, accuracy_train, color='red', label='Train Accuracy')
		axes[0].plot(epochs_history, accuracy_pred, color='blue', label='Prediction Accuracy')
		axes[0].set_title("Accuracy by Epoch")
		axes[0].set_xlabel('Epochs')
		axes[0].set_ylabel('Accuracy')
		axes[0].legend()

		axes[1].plot(epochs_history, loss_history, color='red', label='Train Loss')
		axes[1].plot(epochs_history, loss_pred_history, color='blue', label='Prediction Loss')
		axes[1].set_title("Loss Function by Epoch")
		axes[1].set_xlabel('Epochs')
		axes[1].set_ylabel('Loss')
		axes[1].legend()

		plt.tight_layout()
		plt.show()
