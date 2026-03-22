from collections.abc import Sequence
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import activations
from activations import ActivationPair, ActivationType


class NeuralNetwork:
	"""A fully-connected neural network. Implemented using only numpy."""

	def __init__(
		self,
		structure: Sequence[int],
		activation_types: Sequence[ActivationType] | ActivationType,
		parameters: np.ndarray | None = None,
	) -> None:
		"""
		Initializes a fully-connected neural network.

		args:
			structure: is a list which each element shows the number of neurons in that
				layer. the first element is the input layer.
			parameters(optional): networks pre-trained or desired parameters to be loaded
				to the network
			activations: either a list with len(structure)-1 elements of ActivationType objects
				or a single ActivationType object that will be applied to all layers.

		raises:
			ValueError:
				- if structure has less than 3 elements
				- if parameters is of the wrong size and shape
				- if activation does not have len(structure)-1 elements
		"""

		if len(structure) < 3:
			raise ValueError(
				"The network should have 3 layers minimum! At least 1 hidden layer. "
			)

		self.structure: Sequence[int] = structure

		# number of layers, except the input layer
		self._L: int = len(structure) - 1

		# calculation of the correct NUMBER_OF_PARAMS
		self._calc_num_parameters()

		# setting self._weights_shapes and self._biases_shapes
		self._calc_weight_bias_shapes()

		# if the parameters is passed to __init__
		if parameters is not None:
			# shape is checked in the property.setter
			# this will also update weights and biases
			self.parameters = parameters

		# if parameters is not passed to __init__ set them randomly
		else:
			self._init_rand_weights_biases()

		# checking and setting the activation and their derivative functions
		self.set_activations(activation_types)

		# setting all neuron values to zero
		# setting self.input_layer, self.z_layers, self.layers
		self.init_all_neurons()

	def init_weights_biases(
		self, weights: list[np.ndarray], biases: list[np.ndarray]
	) -> None:
		"""
		This method will take in the weights and biases and set them in the network.
		"""
		if len(weights) != self._L:
			raise ValueError(f"Weights list should contain {self._L} weights matrices!")

		for i, (ws, shape) in enumerate(zip(weights, self._weights_shapes)):
			if ws.shape != shape:
				raise ValueError(
					f"{ws.shape} is a wrong shape.(happened in weights[{i}]) should be {shape}."
				)

		for i, (bs, shape) in enumerate(zip(biases, self._biases_shapes)):
			if bs.shape != shape:
				raise ValueError(
					f"{bs.shape} is a wrong shape.(happened in biases[{i}]) should be {shape}."
				)

		self.weights: list[np.ndarray] = weights
		self.biases: list[np.ndarray] = biases

	def init_all_neurons(self):
		"""Initialize all neurons with zero."""

		self.input_layer: np.ndarray = np.zeros((self.structure[0], 1))

		# to initialize all z, and activations
		self.z_layers: list[np.ndarray] = [
			np.zeros(shape=(self.structure[i + 1], 1)) for i in range(self._L)
		]
		self.layers: list[np.ndarray] = [
			np.zeros(shape=(self.structure[i + 1], 1)) for i in range(self._L)
		]

		self.feed_forward()

	def set_activations(
		self, activation_types: Sequence[ActivationType] | ActivationType
	) -> None:
		"""
		Setting up self.activation_derivative_pairs which holds a list of
		ActivationPair objects.

		args:
			activations: the input that is passed to __init__ which can either be
			a single ActivationType or a list of ActivationType objects

		raises:
			ValueError:
			- if activations is not a single ActivationType or a list of ActivationType objects
			- if activations is a list but its length is not correct and equal to self._L
			- if the list elements are not of type ActivationType

		"""

		activation_types_list: list[ActivationType]

		# check the type of activations
		# if a single ActivationType, apply it to all layers
		# if a list with size L, apply individual activations
		if isinstance(activation_types, ActivationType):
			activation_types_list = [activation_types for _ in range(self._L)]
		elif not isinstance(activation_types, Sequence):
			raise ValueError(
				f"activations should either be a single ActivationType or "
				f"a list of ActivationType objects of size {self._L}."
			)
		else:
			activation_types_list = list(activation_types)

		if len(activation_types_list) != self._L:
			raise ValueError(
				f"activation list should be of size len(structure)-1={self._L}."
				+ f" Got {len(activation_types_list)} instead."
			)

		self.activation_derivative_pairs: list[ActivationPair] = []

		for activation in activation_types_list:
			match activation:
				case ActivationType.Sigmoid:
					self.activation_derivative_pairs.append(
						ActivationPair(f=activations.sigmoid, df=activations.d_sigmoid)
					)
				case ActivationType.Tanh:
					self.activation_derivative_pairs.append(
						ActivationPair(f=activations.tanh, df=activations.d_tanh)
					)
				case ActivationType.Relu:
					self.activation_derivative_pairs.append(
						ActivationPair(f=activations.relu, df=activations.d_relu)
					)
				case ActivationType.NoAct:
					self.activation_derivative_pairs.append(
						ActivationPair(f=lambda x: x, df=lambda x: np.ones(x.shape))
					)
				case _:
					raise ValueError("Activations should be of type ActivationType!")

	@property
	def parameters(self) -> np.ndarray:
		"""This is a full flattened version of self.weights and self.biases."""

		return np.hstack(
			[ws.flatten() for ws in self.weights] + [bs.flatten() for bs in self.biases]
		).reshape((-1, 1))

	@parameters.setter
	def parameters(self, new_params: np.ndarray):
		"""
		This method will set self.parameters property(automatically) and update
		self.weights and self.biases based on it, so that they are always synced.
		"""

		# if the parameters is passed and is of incorrect shape, stop
		if isinstance(new_params, np.ndarray):
			if new_params.shape != (self.NUMBER_OF_PARAMS, 1):
				raise ValueError(
					f"parameters should be of shape {(self.NUMBER_OF_PARAMS, 1)}."
				)

		self.weights = []
		self.biases = []

		# grab the parameters for weights
		count: int = 0
		for shape in self._weights_shapes:
			total = shape[0] * shape[1]
			ws = self.parameters.T[0][count : count + total].reshape(shape)
			self.weights.append(ws)

			count += total

		# grab the parameters for biases
		for shape in self._biases_shapes:
			total = shape[0]
			bs = self.parameters.T[0][count : count + total].reshape(shape)
			self.biases.append(bs)

			count += total

	def _init_rand_weights_biases(self):
		"""Initializes all weights and biases randomly."""

		self.weights = [
			0.05 * np.random.randn(*shape) for shape in self._weights_shapes
		]
		self.biases = [0.01 * np.random.randn(*shape) for shape in self._biases_shapes]

	def _calc_num_parameters(self):
		"""Calculate the total size of parameters."""

		self.NUMBER_OF_PARAMS: int = 0
		for i in range(self._L):
			# number of weights
			self.NUMBER_OF_PARAMS += self.structure[i] * self.structure[i + 1]

			# number of biases
			self.NUMBER_OF_PARAMS += self.structure[i + 1]

	def _calc_weight_bias_shapes(self):
		"""Calculate the shapes of all the weights and biases."""

		self._weights_shapes: list[tuple[int, int]] = []
		self._biases_shapes: list[tuple[int, int]] = []
		# computing the appropriate shape of each weights matrix between layers
		for i in range(self._L):
			self._weights_shapes.append((self.structure[i + 1], self.structure[i]))
			self._biases_shapes.append((self.structure[i + 1], 1))

	def load_input_layer(self, input_vector: np.ndarray) -> None:
		# * input_vector.shape = (self.structure[0], 1)
		if input_vector.shape != (self.structure[0], 1):
			raise ValueError(
				f"input should be of shape {(self.structure[0], 1)}. got {input_vector.shape} instead"
			)

		self.input_layer = input_vector

	def cost_of_single_sample(self, sample: np.ndarray, true_y: np.ndarray) -> float:
		# * input_vector.shape = (self.structure[0], 1)
		if sample.shape != (self.structure[0], 1):
			raise ValueError(
				f"input should be of shape {(self.structure[0], 1)}. got {sample.shape} instead"
			)

		self.load_input_layer(input_vector=sample)
		self.feed_forward()

		# compare the self.layers[-1] and the true output
		# using mean squared error or MSE
		cost = (1 / len(true_y)) * np.sum((self.layers[-1] - true_y) ** 2)
		return cost

	def cost_of_test_data(
		self, test_samples: np.ndarray, true_outputs: np.ndarray
	) -> float:
		# test_samples.shape = (self.structure[0], m)
		# test_samples: is a np array which each col represents one sample
		# true_outputs.shape = (self.structure[-1], m)

		MSE: float = 0
		for sample, output in zip(test_samples.T, true_outputs.T):
			sample = sample.reshape((-1, 1))
			MSE += self.cost_of_single_sample(sample, output)

		M = len(test_samples.T)
		MSE = (1 / M) * MSE
		return MSE

	def accuracy_score(
		self, test_samples: np.ndarray, true_labels: np.ndarray
	) -> float:
		"""
		This method takes a dataset and the corresponding true labels (only the class of the classification problem)
		and outputs the accuracy score.
		it can only be used with classification problems
		"""
		# * test_samples.shape = (self.layer_structure[0], m)
		# * true_labels.shape = (1, m)

		total: int = len(test_samples.T)
		trues: int = 0

		for sample, label in zip(test_samples.T, true_labels[0]):
			sample = sample.reshape((-1, 1))
			result = self.predict_class(sample)
			if result == label:
				trues += 1

		return trues / total


	def predict_output(self, sample: np.ndarray | list) -> np.ndarray:
		"""
		This method takes in a single sample and outputs the entire output layer.
		"""
		if isinstance(sample, list):
			sample = np.array(sample).reshape((-1, 1))

		# * sample.shape = (self.structure[0], 1)
		if sample.shape != (self.structure[0], 1):
			raise ValueError(
				f"{sample.shape} is a bad shape for input."
				+ f" should be {(self.structure[0], 1)}."
			)

		self.load_input_layer(input_vector=sample)
		self.feed_forward()

		return self.layers[-1]

	def predict_class(self, sample: np.ndarray) -> int:
		"""
		This method takes in a single sample and
		returns the index of the highest value in the output layer.
		"""
		output_vector = self.predict_output(sample).flatten()
		return int(np.argmax(output_vector))

	def _backprop_one_sample(self, sample: np.ndarray, output: np.ndarray) -> tuple:
		"""
		This method holds all the math and calculus behind backpropagation
		it calculates the derivitive of the cost w.r.t all the weights and
		biases of the network, for only ONE training data
		inputs:
			sample.shape = (self.structure[0], 1)
			output.shape = (self.structure[-1], 1)
		returns:
			(dw, d)
		"""

		self.load_input_layer(input_vector=sample)
		self.feed_forward()

		# * d_cost_p_ol.shape = (self.structure[-1], 1)
		# * derivative of mean squared error or MSE
		d_cost_p_ol = 2 * (self.layers[-1] - output)

		# * d_activation(z_ol)
		# * times the gradient of the cost w.r.t activations of the output layer
		# * error_ol.shape =(self.structure[-1], 1)
		error_ol = (
			self.activation_derivative_pairs[-1].df(self.z_layers[-1]) * d_cost_p_ol
		)

		# error of all the other layers, except the last layer and the input layer
		# * this errors are gonna be in reverse order, so the first item will be the second to last layer's
		# * and the next will be the third from last layer's and so on ...
		hlayers_errors: list[np.ndarray] = []

		# loop through hidden layers in reverse order, from secnod to last layer, to the second layer
		# L-2 is because we should start at the last hidden layer
		e_count = 0
		for i in range(self._L - 2, -1, -1):
			# * the layer before the output layer
			if i == self._L - 2:
				e = self.activation_derivative_pairs[i].df(self.z_layers[i]) * (
					self.weights[i + 1].T @ error_ol
				)
				hlayers_errors.append(e)
			else:
				# * remember errors[L-2-i] should be used, and it actually means the error of the next layer
				# * this is because it is in the reveresed order
				# e = self.d_activation[i](self.z_layers[i]) * (self.weights[i+1].T @ hlayers_errors[self.L-2-i])
				e = self.activation_derivative_pairs[i].df(self.z_layers[i]) * (
					self.weights[i + 1].T @ hlayers_errors[e_count]
				)
				e_count += 1
				hlayers_errors.append(e)

		# * now we can flip the errors for convenience
		hlayers_errors = hlayers_errors[::-1]

		# of length L
		d_cost_p_biases: list[np.ndarray] = []
		# * based on the equations of backprpoagation we know that d_cost_p_b of each layer
		# * is actually equal to the error of that layer.
		for error in hlayers_errors:
			d_cost_p_biases.append(error)
		d_cost_p_biases.append(error_ol)

		# * based on the equations of backpropagation
		# * the derivative of the cost wr to the weights of the layer l will be
		# * the matrix mult of error of layer l and activation of layer l-1 transposed
		d_cost_p_weights: list[np.ndarray] = []

		d_cost_p_weights.append(hlayers_errors[0] @ self.input_layer.T)
		for i in range(1, self._L - 1):
			d_cost_p_weights.append(hlayers_errors[i] @ self.layers[i - 1].T)
		d_cost_p_weights.append(error_ol @ self.layers[-2].T)

		return (d_cost_p_weights, d_cost_p_biases)

	def backpropagation(self, x_train: np.ndarray, y_train: np.ndarray) -> tuple:
		"""
		This method will run the backprop_one_sample method for a dataset and
		take the average of all the gradients of the weights and biases
		returns:
			(dw, db)
		"""

		# * m training samples
		# * x_train.shape = (self.structure[0], m)
		# * y_train.shape = (self.structure[-1], m)

		# average derivative of cost w.r.t weights
		dw: list[np.ndarray] = [np.zeros(shape) for shape in self._weights_shapes]

		# average derivative of cost w.r.t biases
		db: list[np.ndarray] = [np.zeros(shape) for shape in self._biases_shapes]

		for features, output in zip(x_train.T, y_train.T):
			# reshape the variables to appropriate shapes
			# * output.shape -> (self.structure[-1], 1)
			# * features.shape -> (self.structure[0], 1)
			features = features.reshape((-1, 1))
			output = output.reshape((-1, 1))

			# * label in this method should be an int
			tdw, tdb = self._backprop_one_sample(sample=features, output=output)

			for i in range(self._L):
				dw[i] += tdw[i]
				db[i] += tdb[i]

		# * now each element in the dw and db contain the sum of the derivatives of
		# * the samples inside the training data
		# * now they should be divided by the number of the train sample size, so dw and db, be an average
		train_data_size = x_train.shape[1]

		for i in range(self._L):
			dw[i] /= train_data_size
			db[i] /= train_data_size

		# * now they contain the gradient of the provided dataset
		return (dw, db)

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

	def train(
		self,
		x_train: np.ndarray,
		y_train: np.ndarray,
		*,
		learning_rate: float = 0.1,
		constant_lr: bool = False,
		decay_rate: float = 0.1,
		number_of_epochs: int = 80,
		batch_size: int = 32,
		verbose: bool = False,
	) -> None:
		"""Trains the model with the labeled training data"""

		# * it is assumed that the parameters are initialized randomly

		# * m training samples
		# * x_train.shape = (self.structure[0], m)
		# * y_train.shape = (self.structure[-1], m)
		m = x_train.shape[1]

		# shuffle the training data to avoid biases in the training
		# shuffling along columns
		shuffled_indices = np.random.permutation(m)
		shuffled_x = x_train[:, shuffled_indices]
		shuffled_y = y_train[:, shuffled_indices]

		# * now that the data is shuffled properly
		# * we should divide the data into batches
		num_batches = m // batch_size

		# split column-wise
		batches_x = np.array_split(shuffled_x, num_batches, axis=1)
		batches_y = np.array_split(shuffled_y, num_batches, axis=1)

		initial_lr: float = learning_rate
		for epoch in range(number_of_epochs):
			# * now each batch corresponds to one step at gradient descent
			# * batch_x.shape = (self.structure[0], batch_size), the features
			# * batch_y.shape = (self.structure[-1], batch_size), the true output
			for batch_i, (batch_x, batch_y) in enumerate(zip(batches_x, batches_y)):
				# * the backprop algorithm will run for each batch
				# * one step downhill towards a local minimum
				# * it also updates the self.layers[-1] (output layer)
				# * and consequently the cost, by running self.feed_forward
				dw, db = self.backpropagation(batch_x, batch_y)

				if constant_lr:
					lr = learning_rate
				else:
					lr = np.exp(-epoch * decay_rate) * initial_lr

				# * change each of the weights and biases accordingly
				self._update_parameters(dw, db, learning_rate=lr)

				if verbose and batch_i % 200 == 0:
					print(f"epoch {epoch + 1}:\tbatch={batch_i}/{num_batches}")
			if verbose:
				cost = self.cost_of_test_data(x_train, y_train)
				print(f"\n----\nepoch {epoch + 1}:\tcost = {cost:.4f}")

	def plot_scores(
		self, num_epochs: int, scores: list[float], costs: list[float]
	) -> None:
		epochs_range = list(range(1, num_epochs + 1))

		_, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

		ax1.set_title("Accuracy Score")
		ax1.set_ylabel("Accuracy(%)")
		ax1.set_xlabel("Epoch")
		ax1.plot(epochs_range, scores)

		ax2.set_title("Cost")
		ax2.set_ylabel("Total Cost")
		ax2.set_xlabel("Epoch")
		ax2.plot(epochs_range, costs)

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
				self.z_layers[i] = (self.weights[0] @ self.input_layer) + self.biases[0]
			else:
				self.z_layers[i] = (self.weights[i] @ self.layers[i - 1]) + self.biases[
					i
				]

			self.layers[i] = self.activation_derivative_pairs[i].f(self.z_layers[i])

		if self.layers[-1].shape != (self.structure[-1], 1):
			raise ValueError(
				f"{self.layers[-1].shape} is a bad shape! Should be {(self.structure[-1], 1)}"
			)

	def print_network(self, hidden_layers=False) -> None:
		if hidden_layers:
			# all hidden layers, all layers except output layer
			for i, layer in enumerate(self.layers[:-1]):
				print(f"layer {i}: {layer}")

		print(f"Output layer:\n{self.layers[-1]}")

	def print_stat(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
		score = self.accuracy_score(x_test, y_test)
		cost = self.cost_of_test_data(x_test, y_test)

		print(f"Accuracy = {score * 100:.2f}%")
		print(f"Cost = {cost:.3f}")

	def load_params_from_file(self, filename: str) -> None:
		"""
		load parameters from a saved parameters file
		"""
		with open(filename, "r") as file:
			ps = [float(line.strip()) for line in file.readlines()]

		self.parameters = np.array(ps).reshape((-1, 1))

	def save_parameters_to_file(self, filename: str | None = None) -> None:
		"""
		saves current network parameters to a file
		"""
		if not filename:
			filename = f"parameters_{datetime.now().strftime('%y_%m_%d_%H_%M')}.txt"
		with open(filename, "w") as file:
			for p in self.parameters.reshape((-1,)):
				file.write(f"{p}\n")
