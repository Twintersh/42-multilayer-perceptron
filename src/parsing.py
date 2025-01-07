import configparser
from layers import *

names = {"sigmoid" : Sigmoid, "softmax" : Softmax, "rectifier" : Rectifier}

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
		patience = config.getint("MLP parameters", "patience")
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

	return batch_size, iterate, layers, patience
