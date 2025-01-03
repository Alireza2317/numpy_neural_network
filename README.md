# Neural Networks Implementation From Scratch
This project focuses on the internal mechanics of the neural networks and the basic concepts like:
- what does actually a neuron do?
- how should weights and biases be organized?
- how to train the network?
- how does backpropagation work?
- and much more ...
  
## Getting Started
The very first step would be to create a virtual enviornment and install all the requirements. You can do so by running these commands:

```bash
git clone https://github.com/Alireza2317/neuralnet_from_scratch
cd neuralnet_from_scratch
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
now that you have installed all the dependencies, you can start working with the notebooks.

### Mnist dataset - handwritten digits recognition
This is sort of the 'hello world' of the artificial intelligence realm, so we'll try to use this dataset to train and test the network. `app_mnist.ipynb` notebook is dedicated and designed only for that specific dataset. It's a network with 784 input neurons(since the dataset consists of images with size 28*28 pixels), two internal hidden layers containing 16 neurons each, and finally an output layer with 10 neurons, each corresponding to one digit. The network can be told to use different kinds of activation functions like sigmoid, ReLU and tanh.

This simple network can reach up to 95% accuracy on the test data, which is not that bad. However the problem with this notebook is, the network's design is not at all flexible! The networks structure, meaning the number of layers and the number of neurons in each layer can't be easily changed. and this led me to build a class that handles all that and is super flexible regarding this issue.

### Flexible Neural Network
As mentioned, the constraints of the previous approach led me to create another notebook and basically another python class for a neural network so that it is flexible enough to be used easily to expirement how different changes affect how the neural network learns.
In `app.ipynb` notebook, resides a class which is able to be used easily and change:
- the number of layers
- the number of neurons per each layer
- activation of each layer
- number of epochs in training
- batch size
- learning rate, and wether to use an exponential decay or not

### Thanks
Thank you for checking out this project. Hope you learn something and enjoy :).
