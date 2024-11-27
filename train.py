import pandas as pd
from typing import Tuple
import numpy as np
import os
from layers import Affine, BinaryCrossEntropy, Sigmoid, Softmax
from network import MultilayerPerceptron
from sklearn.preprocessing import StandardScaler

def main():
	(train_data,
	train_label,
	pred_data,
	pred_label) 		= getDataFromDataset("datasets")
	input_size 			= len(train_data)
	hidden_layer_size 	= 20
	l_rate 				= 0.001

	# setting the layers
	input_layer_a = Affine(30, hidden_layer_size, l_rate) # why do the input size is not the len of the batch ???
	input_layer_b = Sigmoid()
	hidden_layer_a = Affine(hidden_layer_size, 2, l_rate)
	hidden_layer_b = Softmax(input_size, 2)
	loss_layer = BinaryCrossEntropy()
	layers = [input_layer_a, input_layer_b, hidden_layer_a, hidden_layer_b]

	# init the MLP with the appropriate layers
	mlp = MultilayerPerceptron(layers, loss_layer, input_size)

	# creating batches 
	# Learning 🧠
	for _ in range(100):
		loss = mlp.calculate_loss(train_data, train_label)
		mlp.backward(train_label)
		print(loss)

def setLabelsValues(label_array: np.array) -> np.array:
	label_mapping = {'M': 1, 'B': 0}
	try:
		return np.array([label_mapping[label] for label in label_array])
	except KeyError:
		exit(42)

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


if __name__ == "__main__":
	main()
