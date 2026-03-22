import random

import numpy as np
import pandas as pd

from NeuralNet import ActivationType, NeuralNetwork


def mnist() -> None:
	train_df = pd.read_csv("./data/mnist_train.csv")
	test_df = pd.read_csv("./data/mnist_test.csv")

	x_train = (
		train_df.iloc[:, 1:].transpose().values
	)  # shape = 784 * m, so each col is a sample
	x_train = x_train / 255.0  # to squish the pixel values between 0-1 instead of 0-255
	y_train = train_df.iloc[:, 0:1].values.reshape((1, -1))  # shape = 1 * m

	x_test = test_df.iloc[:, 1:].values.transpose()
	x_test = x_test / 255.0
	y_test = test_df.iloc[:, 0:1].values.reshape((1, -1))

	np.random.seed(42)
	nn = NeuralNetwork(
		structure=[784, 16, 16, 10],
		activation_types=ActivationType.Relu,
	)

	# nn.print_stat(x_test, y_test)

	nn.train(
		x_train,
		y_train,
		number_of_epochs=100,
		batch_size=32,
		learning_rate=0.5,
		constant_lr=False,
		verbose=True,
	)

	nn.print_stat(x_train, y_train)
	nn.print_stat(x_test, y_test)


def xor() -> None:
	np.random.seed(42)
	random.seed(42)
	nn = NeuralNetwork([2, 3, 1], [ActivationType.Relu, ActivationType.Sigmoid])

	X_train = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

	y_train = np.array([[0, 1, 1, 0]])
	print(X_train.shape)
	print(y_train.shape)

	nn.train(X_train, y_train, verbose=True, batch_size=4, number_of_epochs=2000)

	print()
	print(f"prediction of [0, 0] = {nn.predict_output(np.array([[0], [0]]))[0, 0]:.3f}")
	print(f"prediction of [0, 1] = {nn.predict_output(np.array([[0], [1]]))[0, 0]:.3f}")
	print(f"prediction of [1, 0] = {nn.predict_output(np.array([[1], [0]]))[0, 0]:.3f}")
	print(f"prediction of [1, 1] = {nn.predict_output(np.array([[1], [1]]))[0, 0]:.3f}")


def main() -> None:
	mnist()


if __name__ == "__main__":
	main()
