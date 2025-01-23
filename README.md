# Neural Networks Implementation From Scratch
This project focuses on the internal mechanics and mathematics of neural networks and the basic concepts like:
- what does actually a neuron do?
- how should weights and biases be organized?
- how to train the network?
- how does backpropagation work?
- and much more ...

## Use case
I think this project has 2 important use cases.

1. If you are trying to learn the math behind the implementation of neural networks, this is a great place to start since this project uses nothing but math to build the network.
2. If you need a simple neural network and don't want to install heavy-duty libraries like **tensorflow** or **pytorch**. This simple class can take care of it.


## Getting Started
The very first step would be to create a virtual enviornment and install all the requirements. You can do so by running these commands(on linux/mac):

```bash
git clone https://github.com/Alireza2317/neuralnet_from_scratch
cd neuralnet_from_scratch
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
now that you have installed all the dependencies, you can start working with the notebooks.

### Mnist dataset - handwritten digits recognition
To test the network's performance, we will use the mnist handwritten digits database. This is sort of the *hello world* of the artificial intelligence realm, so we'll try to use this dataset to train and test the network. `app_mnist.ipynb` notebook is dedicated and designed only for that specific dataset. It's a network with 784 input neurons(since the dataset consists of images with size 28*28 pixels), two internal hidden layers containing 16 neurons each, and finally an output layer with 10 neurons, each corresponding to one digit. The network can be adjusted to use different kinds of activation functions like **sigmoid**, **ReLU** and **tanh**.

This simple network can reach up to 97% accuracy on the test data, which is not that bad. However the problem with this notebook is, the network's design is not at all flexible! The networks structure, meaning the number of layers and the number of neurons in each layer can't be easily changed. and this led me to build a class that handles all that and is super flexible regarding this issue.

## Flexible Neural Network
As mentioned, the constraints of the previous approach led me to create another notebook and basically another python class for a neural network so that it is flexible enough to be used easily to expirement how different changes affect how the neural network learns. There is a class in `app.ipynb` notebook and also `NeuralNet.py` file, which is able to be used flexibly. The parameters that are easily changable:
- the number of layers
- the number of neurons in each layer
- activation of each layer
- number of epochs in training
- batch size
- learning rate, and wether to use an exponential decay or not

### Thanks
Thank you for checking out this project. Hope you learn something and enjoy :).
