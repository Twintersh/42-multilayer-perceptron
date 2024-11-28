import pandas as pd
from typing import Tuple
import numpy as np
import os
import random
from layers import Affine, BinaryCrossEntropy, Sigmoid, Softmax
from network import MultilayerPerceptron
from sklearn.preprocessing import StandardScaler

def main():
	(train_data,
	train_label,
	pred_data,
	pred_label) 		= getDataFromDataset("datasets")
	hidden_layer_size 	= 20
	l_rate 				= 1e-3
	batch_size			= 50
	epochs				= len(train_data) * 150

	# setting the layers
	layers = [
		Affine(30, hidden_layer_size, l_rate), # Input layer
		Sigmoid(),
		Affine(hidden_layer_size, hidden_layer_size, l_rate), # hidden layer
		Sigmoid(),
		Affine(hidden_layer_size, 2, l_rate), # Hidden layer
		Softmax(batch_size, 2)
	]
	loss_layer = BinaryCrossEntropy()

	# init the MLP with the appropriate layers
	mlp = MultilayerPerceptron(layers, loss_layer, batch_size)

	for i in range(epochs):
		# creating batches
		rand_index = random.sample(range(len(train_data)), batch_size)
		batch_data = np.array([train_data[j] for j in rand_index])
		batch_label = np.array([train_label[j] for j in rand_index])

		# Learning 🧠
		loss = mlp.calculate_loss(batch_data, batch_label)
		mlp.backward(batch_label)
		if (not (i % 500)):
			print(f"loss: {loss} i: {i}")

	test = mlp.predict(pred_data)
	for i in range(len(test)):
		print(f"{test[i][0]:.1f} {pred_label[i]}")

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


if __name__ == "__main__":
	main()
