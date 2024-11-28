import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from typing import Tuple

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
	train_folder	 = os.path.join(foldername, "train.csv")
	predict_folder	 = os.path.join(foldername, "predict.csv")

	dataset_train	 = pd.read_csv(train_folder, header=None)
	dataset_predict	 = pd.read_csv(predict_folder, header=None)

	tmp_train_label	= dataset_train.to_numpy()[:,1]
	tmp_pred_label	= dataset_predict.to_numpy()[:,1]
	train_label		= setLabelsValues(tmp_train_label)
	pred_label		= setLabelsValues(tmp_pred_label)

	tmp_train_data	= dataset_train.to_numpy()[:,2:]
	tmp_pred_data	= dataset_predict.to_numpy()[:,2:]
	train_data		= normalizeData(tmp_train_data)
	pred_data		= normalizeData(tmp_pred_data)

	return (train_data, train_label, pred_data, pred_label)
