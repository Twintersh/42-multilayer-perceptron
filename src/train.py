from layers import *
from network import MultilayerPerceptron
from parsing import parseConfigFile
from utils import getDataFromDataset, plotGraphs
from predict import getAccuracy
from copy import deepcopy
from io import StringIO
from early_stopping import early_stopping
import numpy as np
import random
import shelve
import argparse
import os

parser = argparse.ArgumentParser(description="A program that trains a multilayer perceptron")
parser.add_argument("-c", "--config", type=str, help="Config file path")
# parser.add_argument("-r", "--seed", type=int, help="Set a random seed")

def train(batch_size, iterate, layers, patience):
	with shelve.open("mlp_material/save_parameters") as save_file:
		(train_data,
		train_label,
		pred_data,
		pred_label) 		= getDataFromDataset("mlp_material")
		epochs				= int(len(train_data) / batch_size) + 1
		loss_history = []
		loss_pred_history = []
		accuracy_train = []
		accuracy_pred = []
		epochs_history = []
		loss_layer = BinaryCrossEntropy()

		print(f"training data shape :\t{train_data.shape}")
		print(f"prediction data shape :\t{pred_data.shape}")

		# init the MLP with the appropriate layers
		mlp = MultilayerPerceptron(layers, loss_layer, batch_size)

		nb_epoch = 0
		for i in range(iterate * epochs):
			# creating batches
			if (batch_size > len(train_data)):
				rand_index = range(len(train_data))
			else:
				rand_index = random.sample(range(len(train_data)), batch_size)
			batch_data = np.array([train_data[j] for j in rand_index])
			batch_label = np.array([train_label[j] for j in rand_index])

			if (batch_size > len(pred_data)):
				rand_index = range(len(pred_data))
			else:
				rand_index = random.sample(range(len(pred_data)), batch_size)
			batch_pred = np.array([pred_data[j] for j in rand_index])
			batch_pred_label = np.array([pred_label[j] for j in rand_index])

			# Learning 🧠
			loss = mlp.calculate_loss(batch_data, batch_label)
			mlp.backward(batch_label)

			# each time i go through the entire dataset (approximation), print loss
			if not i % epochs or i == (iterate * epochs) - 1:
				loss_pred = deepcopy(mlp).calculate_loss(batch_pred, batch_pred_label)
				if not i % (epochs * 100) or i == (iterate * epochs) - 1:
					if early_stopping(loss_pred, patience):
						print("Training was stopped prematurely to avoid overfitting")
						break
					print(f"epoch : {nb_epoch}/{iterate}\tloss: {loss:.4f}\tval_loss: {loss_pred:.4f}")
				nb_epoch += 1
				accuracy_pred.append(getAccuracy(mlp, pred_data, pred_label))
				accuracy_train.append(getAccuracy(mlp, train_data, train_label))
				loss_history.append(loss)
				loss_pred_history.append(loss_pred)
				epochs_history.append(i / epochs)

		save_file["network"] = mlp
		plotGraphs(epochs_history, accuracy_train, accuracy_pred, loss_history, loss_pred_history)


# call main
if __name__ == "__main__":
	args = parser.parse_args()
	if not os.path.exists("mlp_material"):
		print("Error: You must run src/splitDataset.py before training the model.")
		exit(0)
	if args.config:
		batch_size, iterate, layers, patience = parseConfigFile(args.config)
		train(batch_size, iterate, layers, patience)
	else:
		parser.print_help()