import numpy as np
import random
from layers import BinaryCrossEntropy, Sigmoid, Softmax, Layer
from network import MultilayerPerceptron
from utils import getDataFromDataset
import shelve

save_file = shelve.open(".save_parameters")
(train_data,
train_label,
pred_data,
pred_label) = getDataFromDataset(save_file["datasets_dir"])
mlp = save_file["network"]
test = mlp.predict(pred_data)
for i in range(len(test)):
	print(f"{pred_label[i]} : {test[i][0]:.1f}")
