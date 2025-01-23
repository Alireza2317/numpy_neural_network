import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
from datetime import datetime


class NeuralNetwork:
	def __init__(
		self,
		layers_structure: list[int],
		parameters: np.ndarray | None = None,
		activations: list[str] | str = 'sigmoid'
	) -> None:
		if len(layers_structure) < 3:
			raise ValueError(
				'The network should have 3 layers minimum!(input layer + (at least one) hidden layer + output layer.)'
			)
		self.layers_structure: list[int] = layers_structure

		# number of layers, except the input layer
		self._L: int = len(layers_structure) - 1

		# calculation of the correct number of parameters
		self._NUMBER_OF_PARAMS: int = 0
		for i in range(self._L):
			# number of weights
			self._NUMBER_OF_PARAMS += self.layers_structure[i] * self.layers_structure[i+1]

			# number of biases
			self._NUMBER_OF_PARAMS += self.layers_structure[i+1]


		self._weights_shapes: list[tuple[int, int]] = []
		self._biases_shapes: list[tuple[int, int]] = []
		# computing the appropriate shape of each weights matrix between layers
		for i in range(self._L):
			self._weights_shapes.append((self.layers_structure[i+1], self.layers_structure[i]))
			self._biases_shapes.append((self.layers_structure[i+1], 1))


		#? if the parameters is passed to __init__, then set the weights and biases based on that
		if isinstance(parameters, np.ndarray):
			# check the shape before assignment
			if (s := max(parameters.shape)) != self._NUMBER_OF_PARAMS:
				raise ValueError(f'parameters should be of shape ({self._NUMBER_OF_PARAMS}, 1). Got {s} instead.')

			self.parameters: np.ndarray = parameters

			# setting both self.weights and self.biases, based on parameters
			self.parse_parameters()

		#? if parameters is not passed to __init__ set them randomly
		else:
			self.weights: list[np.ndarray] = [np.random.randn(*shape) for shape in self._weights_shapes]
			self.biases: list[np.ndarray] = [np.random.randn(*shape) for shape in self._biases_shapes]

			# sets self.parameters, based on weights and biases
			self.recompute_parameters()


		#* type of activation
		# if only a str, apply it to all layers
		# if a list with size L(number_of_layers except input layer), apply individual activations
		if isinstance(activations, str):
			activations: list[str] = [activations for _ in range(self._L)]

		if len(activations) != self._L:
			raise ValueError(
				f'activation list should be of size [len(layers_structure)-1]={self._L}.\n Got {len(activation)} instead.'
			)

		self.activation_funcs: list[Callable] = []
		self.d_activation_funcs: list[Callable] = []

		for activation in activations:
			match activation:
				case 'sigmoid':
					self.activation_funcs.append(NeuralNetwork._sigmoid)
					self.d_activation_funcs.append(NeuralNetwork._d_sigmoid)
				case 'tanh':
					self.activation_funcs.append(NeuralNetwork._tanh)
					self.d_activation_funcs.append(NeuralNetwork._d_tanh)
				case 'relu':
					self.activation_funcs.append(NeuralNetwork._ReLU)
					self.d_activation_funcs.append(NeuralNetwork._d_ReLU)
				case 'no-activation':
					self.activation_funcs.append(lambda x: x)
					self.d_activation_funcs.append(lambda x: x)
				case _:
					raise ValueError('activations can be a member of ["sigmoid", "relu", "tanh", "no-activation"]')

		# initialize all the neurons with zero
		self.input_layer: np.ndarray = np.zeros((self.layers_structure[0], 1))

		# to initialize all z, and activations
		self.z_layers = [None for _ in range(self._L)]
		self.layers = [None for _ in range(self._L)]

		self.feed_forward()


	def load_input_layer(self, input_vector: np.ndarray) -> None:
		#* input_vector.shape = (self.layers_structure[0], 1)
		if input_vector.shape != (self.layers_structure[0], 1):
			raise ValueError(f'input should be of shape {(self.layers_structure[0], 1)}. got {input_vector.shape} instead')

		self.input_layer = input_vector


	def cost_of_single_sample(self, sample: np.ndarray, true_label: int) -> float:
		#* input_vector.shape = (self.layers_structure[0], 1)
		if sample.shape != (self.layers_structure[0], 1):
			raise ValueError(f'input should be of shape {(self.layers_structure[0], 1)}. got {sample.shape} instead')

		self.load_input_layer(input_vector=sample)
		self.feed_forward()


		# construct the output vector based on the label
		desired_output = np.zeros((self.layers_structure[-1], 1))
		desired_output[true_label] = 1.0

		# compare the self.layers[-1] and the desired_output
		# using mean squared error
		cost = np.sum((self.layers[-1] - desired_output)**2)
		return cost


	def cost_of_test_data(self, test_samples: np.ndarray, true_labels: np.ndarray) -> float:
		# samples.shape = (self.layers_structure[0], m)
	 	# true_labels.shape = (1, m)
		# samples: is a np array which each col represents one sample

		MSE: float = 0
		for sample, label in zip(test_samples.T, true_labels[0]):
			sample = sample.reshape((-1, 1))
			MSE += self.cost_of_single_sample(sample, label)

		M = len(test_samples.T)
		MSE = (1 / M) * MSE
		return MSE


	def accuracy_score(self, test_samples: np.ndarray, true_labels: np.ndarray) -> float:
		"""
		This method takes a dataset and the corresponding true labels (only the class of the classification problem)
		and outputs the accuracy score.
		it can only be used with classification problems
		"""
		#* test_samples.shape = (self.layer_structure[0], m)
		#* true_labels.shape = (1, m)

		total: int = len(test_samples.T)
		trues: int = 0

		for sample, label in zip(test_samples.T, true_labels[0]):
			sample = sample.reshape((-1, 1))
			result = self.predict_class(sample)
			if result == label:
				trues += 1

		return (trues / total)


	def predict_output(self, sample: np.ndarray) -> np.ndarray:
		"""
		This method takes in a single sample and outputs the entire output layer.
		"""
		#* sample.shape = (self.layers_structure[0], 1)
		if sample.shape != (self.layers_structure[0], 1):
			raise ValueError(f'{sample.shape} is a bad shape for input. should be {(self.layers_structure[0], 1)}.')

		self.load_input_layer(input_vector=sample)
		self.feed_forward()

		return self.layers[-1]


	def predict_class(self, sample: np.ndarray) -> int:
		"""
		This method takes in a single sample and outputs the index of the highest value in the output layer.
		"""
		output_vector = self.predict_output(sample).flatten()
		return np.argmax(output_vector)


	def _backprop_one_sample(self, sample: np.ndarray, label: int) -> tuple:
		"""
		This method holds all the math and calculus behind backpropagation
		it calculates the derivitive of the cost w.r.t all the weights and
		biases of the network, for only ONE training data
		"""

		self.load_input_layer(input_vector=sample)
		self.feed_forward()

		#* convert the label in int format into a one-hot vector
		desired_output = np.zeros(self.layers[-1].shape)
		desired_output[label] = 1.0

		#* d_cost_p_ol.shape = (self.layers_structure[-1], 1)
		#* derivative of mean squared error
		d_cost_p_ol = 2 * (self.layers[-1] - desired_output)

		#* d_activation(z_ol)
		#* times the gradient of the cost w.r.t activations of the output layer
		#* error_ol.shape =(self.layers_structure[-1], 1)
		error_ol = self.d_activation_funcs[-1](self.z_layers[-1]) *  d_cost_p_ol

		# error of all the other layers, except the last layer and the input layer
		#* this errors are gonna be in reverse order, so the first item will be the second to last layer's
		#* and the next will be the third from last layer's and so on ...
		hlayers_errors: list[np.ndarray] = []


		# loop through hidden layers in reverse order, from secnod to last layer, to the second layer
		# L-2 is because we should start at the last hidden layer
		e_count = 0
		for i in range(self._L - 2, -1, -1):
			#* the layer before the output layer
			if i == self._L - 2:
				e = self.d_activation_funcs[i](self.z_layers[i]) * (self.weights[i+1].T @ error_ol)
				hlayers_errors.append(e)
			else:
				#* remember errors[L-2-i] should be used, and it actually means the error of the next layer
				#* this is because it is in the reveresed order
				#e = self.d_activation[i](self.z_layers[i]) * (self.weights[i+1].T @ hlayers_errors[self.L-2-i])
				e = self.d_activation_funcs[i](self.z_layers[i]) * (self.weights[i+1].T @ hlayers_errors[e_count])
				e_count += 1
				hlayers_errors.append(e)

		#* now we can flip the errors for convenience
		hlayers_errors = hlayers_errors[::-1]

		# of length L
		d_cost_p_biases: list[np.ndarray] = []
		#* based on the equations of backprpoagation we know that d_cost_p_b of each layer
		#* is actually equal to the error of that layer.
		for error in hlayers_errors:
			d_cost_p_biases.append(error)
		d_cost_p_biases.append(error_ol)


		#* based on the equations of backpropagation
		#* the derivative of the cost wr to the weights of the layer l will be
		#* the matrix mult of error of layer l and activation of layer l-1 transposed
		d_cost_p_weights: list[np.ndarray] = []

		d_cost_p_weights.append(hlayers_errors[0] @ self.input_layer.T)
		for i in range(1, self._L - 1):
			d_cost_p_weights.append(error[i] @ self.layers[i-1].T)
		d_cost_p_weights.append(error_ol @ self.layers[-2].T)


		return (d_cost_p_weights, d_cost_p_biases)


	def backpropagation(self, x_train: np.ndarray, y_train: np.ndarray) -> tuple:
		"""
		This method will run the backprop_one_sample method for a dataset and
		take the average of all the gradients of the weights and biases
		"""

		#* m training samples
		#* x_train.shape = (self.layers_structure[0], m)
		#* y_train.shape = (1, m)

		# average derivative of cost w.r.t weights
		dw: list[np.ndarray] = [np.zeros(shape) for shape in self._weights_shapes]

		# average derivative of cost w.r.t biases
		db: list[np.ndarray] = [np.zeros(shape) for shape in self._biases_shapes]

		for features, label in zip(x_train.T, y_train[0]):
			#* label: int
			#* features.shape = (self.layers_structure[0], 1)
			features = features.reshape((-1, 1))

			#* label in this method should be an int
			tdw, tdb = self._backprop_one_sample(sample=features, label=label)

			for i in range(self._L):
				dw[i] += tdw[i]
				db[i] += tdb[i]

		#* now each element in the dw and db contain the sum of the derivatives of
		#* the samples inside the training data
		#* now they should be divided by the number of the train sample size, so dw and db, be an average
		train_data_size = x_train.shape[1]

		for i in range(self._L):
			dw[i] /= train_data_size
			db[i] /= train_data_size

		#* now they contain the gradient of the provided dataset
		return (dw, db)


	def recompute_parameters(self) -> None:
		"""
		This method will recompute self.parameters from the current self.weights and self.biases
		"""
		self.parameters = np.array([])
		# first the weights
		for ws in self.weights:
			self.parameters = np.append(self.parameters, ws.flatten())
		# then the biases
		for bs in self.biases:
			self.parameters = np.append(self.parameters, bs.flatten())


		self.parameters = self.parameters.reshape((-1, 1))


	def init_weights_biases(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
		"""
		This method will take in the weights and biases and sets them in the network.
		"""
		if len(weights) != self._L:
			raise ValueError(f'Weights list should contain {self._L} weights matrices!')

		for i, (ws, shape) in enumerate(zip(weights, self._weights_shapes)):
			if ws.shape != shape:
				raise ValueError(f'{ws.shape} is a wrong shape.(happened in weights[{i}]) should be {shape}.')

		for i, (bs, shape) in enumerate(zip(biases, self._biases_shapes)):
			if bs.shape != shape:
				raise ValueError(f'{bs.shape} is a wrong shape.(happened in biases[{i}]) should be {shape}.')

		self.weights: list[np.ndarray] = weights
		self.biases: list[np.ndarray] = biases
		self.recompute_parameters()


	def _update_parameters(self, dw, db, learning_rate) -> None:
		"""
		This method will update the weights and biases based on the given gradients and learning rate.
		basically applying gradient descent
		"""
		self.weights = [
			weights_matrix - (learning_rate * dw[i])
			for i, weights_matrix in enumerate(self.weights)
		]

		self.biases = [
			biases_vector - (learning_rate * db[i])
			for i, biases_vector in enumerate(self.biases)
		]

		#* now self.parameters, which basically is the flattened version of
		#* all the weights and biases, should be updated as well
		self.recompute_parameters()


	def parse_parameters(self, parameters: np.ndarray | None = None) -> None:
		"""
		This method will parse self.parameters or the given parameters and set each parameter to
		the corresponding weights and biases.
		"""
		# if the parameters is passed and is of incorrect shape, stop
		if isinstance(parameters, np.ndarray):
			if parameters.shape != (self._NUMBER_OF_PARAMS, 1):
				raise ValueError(f'parameters should be of shape {(self._NUMBER_OF_PARAMS, 1)}.')

			# update parameters
			self.parameters = parameters

		# otherwise use the current self.parameters

		self.weights: list[np.ndarray] = []
		self.biases: list[np.ndarray] = []

		# grab the parameters for weights
		count: int = 0
		for shape in self._weights_shapes:
			total = shape[0] * shape[1]
			ws = self.parameters.T[0][count:count+total].reshape(shape)
			self.weights.append(ws)

			count += total

		# grab the parameters for biases
		for shape in self._biases_shapes:
			total = shape[0]
			bs = self.parameters.T[0][count:count+total].reshape(shape)
			self.biases.append(bs)

			count += total


	def train(
			self,
			x_train: np.ndarray,
			y_train: np.ndarray,
			*,
			learning_rate: float = 0.1,
			constant_lr: bool = False,
			decay_rate: float = 0.1,
			number_of_epochs: int = 80,
			mini_batches_size: int = 100,
			quiet: bool = False
	) -> None:
		"""Trains the model with the labeled training data"""

		if x_train.shape[1] > mini_batches_size:
			raise ValueError(
				f'Batch size should be <= to the number of training samples!'
			)

		#* initialize the parameters randomly
		#* it is assumed that the parameters are initialized randomly

		#* m training samples
		#* x_train.shape = (self.layers_structure[0], m)
		#* y_train.shape = (1, m)

		#* first we'd better attach the x_train and y_train together
		#* then we can shuffle the training data
		#* and then seperate the x and y again
		#* add y row to the 0-th row of train_data
		train_data = np.vstack((y_train, x_train))

		#! now because shuffle, shuffles the array based on the rows
		#! but we need to shuffle the data based on the coloumns
		#! we have to transpose it twice to get around this
		train_data = train_data.T
		np.random.shuffle(train_data)
		train_data = train_data.T

		#* now that the data is shuffled properly
		#* we should divide the data into mini-batches
		mini_batches: list[np.ndarray] = [
			train_data[:, count:count+mini_batches_size]
			for count in range(0, len(train_data.T), mini_batches_size)
		]


		#* keep track of the accuracy scores and costs, to plot later
		if not quiet:
			scores: list[float] = []
			costs: list[float] = []

		initial_lr: float = learning_rate
		for epoch in range(number_of_epochs):
			#* now each mini-batch corresponds to one step at gradient descent
			#* batch.shape = (self.layers_structure[0]+1, mini_batches_size), the label and the features
			for batch in mini_batches:
				#* batch.shape = (self.layers_structure[0]+1, mini_batches_size)
				x_train_batch = batch[1:] # (self.layers_structure[0], mini_batches_size)
				y_train_batch = batch[0:1] # (1, mini_batches_size)

				# now because y_train_batch holds the labels, and they should be int
				# int8 is enough because we know labels are single digit ints
				y_train_batch = y_train_batch.astype(np.int8)

				#* the backprop algorithm will run for each batch, one step downhill towards a local minima
				#* it also updates the self.layers[-1](output layer)
				#* and consequently the cost, by running self.feed_forward
				dw, db = self.backpropagation(x_train_batch, y_train_batch)

				if constant_lr:
					lr = learning_rate
				else:
					lr = np.exp(-epoch * decay_rate) * initial_lr


				#* change each of the weights and biases accordingly
				self._update_parameters(dw, db, learning_rate=lr)

			#* parameters are updated now
			if quiet: continue

			score = self.accuracy_score(x_train, y_train)*100
			cost = 	self.cost_of_test_data(x_train, y_train)
			scores.append(score)
			costs.append(cost)
			print(f'epoch {epoch+1}:\taccuracy = {score:.2f}%\tcost = {cost:.4f}')

		if quiet: return
		self.plot_scores(number_of_epochs, scores, costs)


	def plot_scores(self, num_epochs: int, scores: list[float], costs: list[float]) -> None:
		epochs_range =  list(range(1, num_epochs+1))

		_, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

		ax1.set_title('Accuracy Score')
		ax1.set_ylabel('Accuracy(%)')
		ax1.set_xlabel('Epoch')
		ax1.plot(epochs_range, scores)

		ax2.set_title('Cost')
		ax2.set_ylabel('Total Cost')
		ax2.set_xlabel('Epoch')
		ax2.plot(epochs_range, costs)


	@staticmethod
	def _sigmoid(z: np.ndarray) -> np.ndarray:
		return 1 / (1 + np.exp(-z))

	@staticmethod
	def _d_sigmoid(z: np.ndarray) -> np.ndarray:
		return np.exp(-z) / (np.pow((1 + np.exp(-z)), 2))


	@staticmethod
	def _ReLU(z: np.ndarray) -> np.ndarray:
		return np.maximum(0, z)


	@staticmethod
	def _d_ReLU(z: np.ndarray) -> np.ndarray:
		return (z > 0).astype(np.float64)


	@staticmethod
	def _tanh(z: np.ndarray) -> np.ndarray:
		return np.tanh(z)


	@staticmethod
	def _d_tanh(z: np.ndarray) -> np.ndarray:
		return 4 * np.exp(2 * z) / np.power(np.exp(2*z) + 1, 2)


	@staticmethod
	def _softmax(z: np.ndarray) -> np.ndarray:
		return np.exp(-z) / np.sum(np.exp(-z))


	def feed_forward(self) -> None:
		"""
		Will calculate all the values in all the layers
		based on the weights and biases
		"""

		# loop through each layer(except input layer) and calculate the z of the next layer
		# and the activation of the current
		for i in range(self._L):
			# connection between the input layer and the next layer
			if i == 0:
				self.z_layers[i] = (self.weights[0] @  self.input_layer) + self.biases[0]
			else:
				self.z_layers[i] = (self.weights[i] @ self.layers[i-1]) + self.biases[i]

			self.layers[i] = self.activation_funcs[i](self.z_layers[i])


		if self.layers[-1].shape != (self.layers_structure[-1], 1):
			raise ValueError(f'{self.layers[-1].shape} is a bad shape! Should be {(self.layers_structure[-1], 1)}')


	def print_network(self, hidden_layers = False) -> None:
		if hidden_layers:
			# all hidden layers, all layers except output layer
			for i, layer in enumerate(self.layers[:-1]):
				print(f'layer {i}: {layer}')

		print(f'Output layer:\n{self.layers[-1]}')


	def print_stat(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
		score = self.accuracy_score(x_test, y_test)
		cost = self.cost_of_test_data(x_test, y_test)

		print(f'Accuracy = {score * 100:.2f}%')
		print(f'Cost = {cost:.3f}')


	def	load_params_from_file(self, filename: str) -> None:
		"""
		load parameters from a saved parameters file
		"""
		with open(filename, 'r') as file:
			ps = [float(line.strip()) for line in file.readlines()]

		self.parse_parameters(np.array(ps).reshape((-1, 1)))


	def save_parameters_to_file(self, filename: str | None = None) -> None:
		"""
		saves current network parameters to a file
		"""
		if not filename:
			filename = f"parameters_{datetime.now().strftime('%y_%m_%d_%H_%M')}.txt"
		with open(filename, 'w') as file:
			for p in self.parameters.reshape((-1, )):
				file.write(f'{p}\n')
