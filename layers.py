import numpy as np

class Affine:
	def __init__(self, input_size, output_size, l_rate):
		self.x = None
		self.w = np.random.randn(input_size, output_size) / np.sqrt(input_size) # ?
		self.b = np.zeros(output_size) # ?
		self.input_size = input_size
		self.output_size = output_size
		self.lr_rate = l_rate

	def forward(self, x):
		self.x = x # too big values
		return np.dot(x, self.w) + self.b

	def backward(self, dx):
		dw = np.dot(self.x.T, dx)
		db = np.sum(dx, axis=0)
		self.w = self.w - np.dot(self.lr_rate, dw)
		self.b = self.b - np.dot(self.lr_rate, db)
		# calculates the loss
		res = np.dot(dx, self.w.T)
		return (res)

	# def save_parameters():

class Sigmoid:
	def __init__(self):
		self.out = None

	def forward(self, x):
		res = 1 / (1 + np.exp(-x))
		self.out = res
		return (res)

	def backward(self, dx):
		res = dx * (1 - self.out) * self.out
		return res

	# def save_parameters():

class Softmax:
	def __init__(self, input_size, output_size):
		self.x = None
		self.dx = np.zeros((input_size, output_size))
		self.input_size = input_size
		self.output_size = output_size

	def forward(self, x):
		exp_sum = np.sum(np.exp(x), axis=1)
		for i in range(x.shape[0]):
			x[i, :] = np.exp(x[i, :]) / exp_sum[i]
		self.x = x
		return x

	def backward(self, label):
		print(len(label))
		for i in range(self.x.shape[0]):
			if label[i] == 1:
				self.dx[i] = np.array([self.x[i, 0] - 1, self.x[i, 1] - 0])
			else:
				self.dx[i] = np.array([self.x[i, 0] - 0, self.x[i, 1] - 1])
			print(i)
		print(len(self.dx))
		exit(42)
		return self.dx
	# def save_parameters(self, params):

class BinaryCrossEntropy:
	def forward(self, x, label):
		loss_sum = 0
		for i in range(x.shape[0]):
			prob = x[i, :][0]
			loss_sum += label[i] * np.log(prob) + (1 - label[i]) * np.log(1 - prob)

		return (-1) * loss_sum / x.shape[0]

	# def save_parameters(self, params):
	#     param_dict = {}
	#     param_dict["layer"] = "BinaryCrossEntropy"
	#     params.append(param_dict)
	#     return params
