import pandas as pd
from typing import Tuple
import numpy as np
import os
from layers import Affine, BinaryCrossEntropy, Sigmoid, Softmax
from network import MultilayerPerceptron

def main():
	(train_data,
	train_label,
	pred_data,
	pred_label) 		= getData("datasets")
	input_size 			= len(train_data)
	hidden_layer_size 	= 50
	l_rate 				= 0.001

	input_layer_a = Affine(30, hidden_layer_size, l_rate) # why do the input size is not the len of the batch ???
	input_layer_b = Sigmoid()
	hidden_layer_a = Affine(hidden_layer_size, 2, l_rate)
	hidden_layer_b = Softmax(input_size, 2)
	loss_layer = BinaryCrossEntropy()
	layers = [input_layer_a, input_layer_b, hidden_layer_a, hidden_layer_b]
	mlp = MultilayerPerceptron(layers, loss_layer, input_size)
	print(f"{len(train_data)}:::{len(train_label)}")
	loss1 = mlp.calculate_loss(train_data, train_label)
	print("first loss calculated.")
	mlp.backward(train_label)
	loss2 = mlp.calculate_loss(train_data, train_label)
	print(loss1, loss2)

def getData(foldername: str) -> Tuple[np.array, np.array, np.array, np.array]:
	train_folder	 = os.path.join(foldername, "train.csv")
	predict_folder	 = os.path.join(foldername, "predict.csv")

	dataset_train	 = pd.read_csv(train_folder, header=None)
	dataset_predict	 = pd.read_csv(predict_folder, header=None)

	train_data = dataset_train.to_numpy()[:,2:]
	train_label = dataset_train.to_numpy()[:,1]
	pred_data = dataset_predict.to_numpy()[:,2:]
	pred_label = dataset_predict.to_numpy()[:,1]
	return (train_data, train_label, pred_data, pred_label)


if __name__ == "__main__":
	main()
