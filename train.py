from layers import *
from network import MultilayerPerceptron
from utils import getDataFromDataset
from predict import getAccuracy
from copy import deepcopy
from io import StringIO
import numpy as np
import random
import matplotlib.pyplot as plt
import shelve
import configparser
import argparse

names = {"sigmoid" : Sigmoid, "softmax" : Softmax, "rectifier" : Rectifier}

parser = argparse.ArgumentParser(description="A program that trains a multilayer perceptron")
parser.add_argument("-c", "--config", type=str, help="Config file path")
# parser.add_argument("-r", "--seed", type=int, help="Set a random seed")

def parseConfigFile(filename):
	layers = []
	config = configparser.ConfigParser()

	# pre-process the file to handle newlines
	with open(filename, 'r') as file:
		content = file.read().replace('\\\n', ';')

	# get the values in the config file
	try:
		config.read_string(content)
		# get the optional seed value in the config file
		if config.has_option("MLP parameters", "seed"):
			print()
			try:
				seed = config.getint("MLP parameters", "seed")
				np.random.seed(seed)
				random.seed(seed)
				print(f"Random seed set to {seed}")
			except ValueError:
				print("Error: 'seed' in the configuration must be an integer.")

		hidden_layer_size = config.getint("MLP parameters", "hidden_layer_size")
		l_rate = config.getfloat("MLP parameters", "l_rate")
		batch_size = config.getint("MLP parameters", "batch_size")
		iterate = config.getint("MLP parameters", "iterate")
		activation_functions = [layer.strip().replace('"', '') for layer in config.get("MLP parameters", "layers").split(';') if not (layer.strip() == '"' or layer.strip() == '')]

		print(f"Neural network shape: {activation_functions}")

		# change the activation_function's strings to actual layer objects
		for activation_function in activation_functions:
			activation_function = activation_function.lower()
			if not activation_function in names:
				exit("bad activation function name given to layers")
			else:
				layers.append(Layer(
					ActivationFunction = names[activation_function],
					output_size = 2 if activation_function == "softmax" else hidden_layer_size,
					l_rate = l_rate
				))

	except (configparser.NoOptionError, SystemExit) as e:
		print(f"Error parsing the configuration file: {e}")
		exit(1)

	return batch_size, iterate, layers


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


def train(batch_size, iterate, layers):
	with shelve.open(".save_parameters") as save_file:
		(train_data,
		train_label,
		pred_data,
		pred_label) 		= getDataFromDataset("datasets")
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
			loss_pred = deepcopy(mlp).calculate_loss(batch_pred, batch_pred_label)

			# implement early stopping here

			mlp.backward(batch_label)
			# each time i go through the entire dataset (approximation), print loss
			if not i % epochs or i == (iterate * epochs) - 1:
				if not i % (epochs * 100) or i == (iterate * epochs) - 1:
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
	if args.config:
		batch_size, iterate, layers = parseConfigFile(args.config)
		train(batch_size, iterate, layers)
	else:
		parser.print_help()