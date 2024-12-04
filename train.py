import numpy as np
import random
from layers import BinaryCrossEntropy, Sigmoid, Softmax, Layer
from network import MultilayerPerceptron
from utils import getDataFromDataset
import matplotlib.pyplot as plt
import shelve
from copy import deepcopy

def train():
	with shelve.open(".save_parameters") as save_file:
		(train_data,
		train_label,
		pred_data,
		pred_label) 		= getDataFromDataset("datasets")
		hidden_layer_size 	= 30
		l_rate 				= 1e-3
		batch_size			= 100
		iterate				= 500 # number of times we go through the dataset
		epochs				= int(len(train_data) / batch_size) + 1
		loss_history = []
		loss_pred_history = []
		epochs_history = []

		print(f"training data shape :\t{train_data.shape}")
		print(f"prediction data shape :\t{pred_data.shape}")

		# setting the layers
		layers = [
			Layer(Sigmoid, hidden_layer_size, l_rate),
			Layer(Sigmoid, hidden_layer_size, l_rate),
			Layer(Softmax, 2, l_rate),
		]
		loss_layer = BinaryCrossEntropy()

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
			mlp.backward(batch_label)
			# each time i go through the entire dataset (approximation), print loss
			if not i % epochs or i == (iterate * epochs) - 1:
				if not i % (epochs * 100) or i == (iterate * epochs) - 1:
					print(f"epoch : {nb_epoch}/{iterate}\tloss: {loss:.4f}\tval_loss: {loss_pred:.4f}")
				nb_epoch += 1
				loss_history.append(loss)
				loss_pred_history.append(loss_pred)
				epochs_history.append(i / epochs)

		save_file["network"] = mlp
		plt.plot(epochs_history, loss_history, color='red')
		plt.plot(epochs_history, loss_pred_history, color='b')
		plt.title("loss function by epoch")
		plt.xlabel('epochs')
		plt.ylabel('Loss')
		plt.show()


# call main
if __name__ == "__main__":
	train()
