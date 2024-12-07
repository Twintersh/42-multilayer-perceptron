import numpy as np

class Layer:
	def __init__(self, ActivationFunction, output_size, l_rate):
		self.affine = Affine(output_size, l_rate)
		self.activation_function = ActivationFunction()

	def forward(self, x):
		res = self.affine.forward(x)
		return self.activation_function.forward(res)

	def backward(self, dx):
		res = self.activation_function.backward(dx)
		return self.affine.backward(res)


class ActivationFunction:
	def forward (self, x):
		# need to be set up in other function
		return x

	def backward (self, dx):
		# need to be set up in other function
		return dx


class Affine(ActivationFunction):
	def __init__(self, output_size, l_rate):
		self.x = None
		self.w = None
		self.b = np.zeros(output_size) # ?
		self.lr_rate = l_rate
		self.output_size = output_size

	def forward(self, x):
		self.x = x
		input_size = x.shape[1]
		if self.w is None:
			self.w = np.random.randn(input_size, self.output_size) / np.sqrt(input_size)
		return np.dot(x, self.w) + self.b

	def backward(self, dx):
		dw = np.dot(self.x.T, dx)
		db = np.sum(dx, axis=0)
		self.w = self.w - np.dot(self.lr_rate, dw)
		self.b = self.b - np.dot(self.lr_rate, db)
		# calculates the loss
		res = np.dot(dx, self.w.T)
		return (res)


class Sigmoid(ActivationFunction):
	def __init__(self):
		self.out = None

	def forward(self, x):
		res = 1 / (1 + np.exp(-x))
		self.out = res
		return (res)

	def backward(self, dx):
		res = dx * (1 - self.out) * self.out
		return res


class Softmax(ActivationFunction):
	def __init__(self):
		self.x = None
		self.dx = None

	def forward(self, x):
		self.dx = np.zeros((x.shape[0], x.shape[1]))

		exp_sum = np.sum(np.exp(x), axis=1)
		for i in range(x.shape[0]):
			x[i, :] = np.exp(x[i, :]) / exp_sum[i]
		self.x = x
		return x

	def backward(self, label):
		for i in range(self.x.shape[0]):
			if label[i] == 1:
				self.dx[i] = np.array([self.x[i, 0] - 1, self.x[i, 1] - 0])
			else:
				self.dx[i] = np.array([self.x[i, 0] - 0, self.x[i, 1] - 1])
		return self.dx


class Rectifier(ActivationFunction):
	def __init__(self):
		self.x = None

	def forward(self, x):
		self.x = x
		return np.maximum(0, x)

	def backward(self, label):
		return label *(self.x > 0).astype(float)


class BinaryCrossEntropy:
	def forward(self, x, label):
		loss_sum = 0
		for i in range(x.shape[0]):
			prob = x[i, :][0]
			loss_sum += label[i] * np.log(prob) + (1 - label[i]) * np.log(1 - prob)

		return (-1) * loss_sum / x.shape[0]
