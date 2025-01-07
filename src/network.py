import shelve

class MultilayerPerceptron:
	def __init__(self, layers, loss_layer, batch_size):
		self.layers = layers
		self.loss_layer = loss_layer
		self.batch_size = batch_size

	def predict(self, x):
		input_arr = x
		for layer in self.layers:
			res = layer.forward(input_arr)
			input_arr = res
		return res

	def calculate_loss(self, x, label):
		res = self.predict(x)
		return self.loss_layer.forward(res, label)

	def backward(self, label):
		dx = label
		for layer in reversed(self.layers):
			res = layer.backward(dx)
			dx = res


