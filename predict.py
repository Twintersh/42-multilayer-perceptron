import numpy as np
import random
from layers import BinaryCrossEntropy, Sigmoid, Softmax, Layer
from network import MultilayerPerceptron
from utils import getDataFromDataset
import shelve

def getAccuracy(mlp, data, label):
	pred = mlp.predict(data)
	nb_correct_pred = 0
	for i in range(len(pred)):
		tmp = float(label[i]) - pred[i][0]
		if abs(tmp) < 0.5:
			nb_correct_pred += 1.0
	return nb_correct_pred / float(len(label))


def predict():
	try:
		save_file = shelve.open(".save_parameters")
		(train_data,
		train_label,
		val_data,
		val_label) = getDataFromDataset("datasets")
		mlp = save_file["network"]
	except Exception as e:
		print("Error: Please run the train program first")
		exit(1)

	accuracy_val = getAccuracy(mlp, val_data, val_label)
	accuracy_train = getAccuracy(mlp, train_data, train_label)

	width = max(len(str(train_data.shape)), len(str(val_data.shape))) + 4
	print("=" * (width + 21))
	print(f"| DATA FORMAT:".ljust(width + 20) + "|")
	print("|".ljust(width + 20) + "|")
	print(f"| train data     : {train_data.shape}".ljust(width + 20) + "|")
	print(f"| validation data: {val_data.shape}".ljust(width + 20) + "|")
	print("=" * (width + 21))
	print(f"| accuracy_train : {accuracy_train:.4f}".ljust(width + 20) + "|")
	print(f"| accuracy_val   : {accuracy_val:.4f}".ljust(width + 20) + "|")
	print("=" * (width + 21))


if __name__ == "__main__":
	predict()