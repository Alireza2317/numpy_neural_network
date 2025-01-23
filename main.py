import pandas as pd
import numpy as np
from NeuralNet import NeuralNetwork


def main():
	train_df = pd.read_csv('./data/mnist_train.csv')
	test_df = pd.read_csv('./data/mnist_test.csv')

	x_train = train_df.iloc[:, 1:].transpose().values # shape = 784 * m, so each col is a sample
	x_train = (x_train / 255.0) # to squish the pixel values between 0-1 instead of 0-255
	y_train = train_df.iloc[:, 0:1].values.reshape((1, -1)) # shape = 1 * m

	x_test = test_df.iloc[:, 1:].values.transpose()
	x_test = (x_test / 255.0)
	y_test = test_df.iloc[:, 0:1].values.reshape((1, -1))


	np.random.seed(42)
	ps = np.random.randn(13002, 1)
	NN = NeuralNetwork(layers_structure=[784, 16, 16, 10], parameters=ps, activations='sigmoid')
	NN.parse_parameters(parameters=ps)

	NN.print_stat(x_test, y_test)

	NN.train(x_train, y_train, number_of_epochs=3, mini_batches_size=120, learning_rate=1, constant_lr=False)

	NN.print_stat(x_train, y_train)
	NN.print_stat(x_test, y_test)



if __name__ == '__main__':
	main()
