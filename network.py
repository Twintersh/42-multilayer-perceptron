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

	# def save_parameters(self, output_param_path):
	# 	params = []
	# 	for layer in self.layers:
	# 		params = layer.save_parameters(params)

	# 	params = self.loss_layer.save_parameters(params)
	# 	save_data = {}
	# 	save_data["params"] = params
	# 	save_data["batch_size"] = self.batch_size
	# 	with open(output_param_path, "w") as f:
	# 		json.dump(save_data, f, indent=4)


