# NumPy Neural Network: Deep Learning from Scratch 🧠

![Python](https://img.shields.io/badge/python-gray?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy)
![License](https://img.shields.io/badge/license-MIT-grey)

A flexible, lightweight Deep Learning framework built entirely from scratch using **Python and NumPy**. No TensorFlow, no PyTorch. Just raw mathematics.

This project focuses on the internal mechanics of Artificial Intelligence, answering fundamental questions such as:
* **What actually is a neuron?**
* **How should weights and biases be organized** in matrix form?
* **How does the network learn** via backpropagation?
* **How does the gradient descent algorithm** minimize error?

## 🎯 Use Cases
This project serves two main purposes:

1.  **Educational Deep Dive:** If you are trying to learn the math behind Neural Networks, this is the perfect starting point. The code relies solely on linear algebra and calculus, exposing the "black box" logic often hidden by high-level libraries.
2.  **Lightweight Implementation:** If you need a simple neural network for a project and don't want the overhead of installing heavy-duty libraries like **TensorFlow** or **PyTorch**, this `NeuralNetwork` class provides a dependency-free solution.

## ✨ Features: The Flexible Network
To overcome the limitations of hard-coded tutorials, I developed a modular `NeuralNetwork` class (found in `NeuralNet.py` and `app.ipynb`) that allows for rapid experimentation.

You can easily customize:
* **Network Topology:** The number of layers and neurons per layer.
* **Activations:** Sigmoid, ReLU, Tanh.
* **Hyperparameters:**
    * Learning Rate (with optional exponential decay)
    * Number of Epochs
    * Batch Size

## 📊 Performance: MNIST Digit Recognition
To validate the engine, the network was tested on the **MNIST handwritten digits dataset** (the "Hello World" of AI).

* **Architecture:** `[784 Input] -> [16 Hidden] -> [16 Hidden] -> [10 Output]`
* **Result:** The network achieves **~97% accuracy** on test data.
* **Demo:** See `app_mnist.ipynb` for the full training loop on image data.

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/Alireza2317/numpy-neural-network
cd numpy-neural-network

# Create virtual env (Recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Usage API
The core logic resides in `NeuralNet.py`. You can use it in your own scripts like this:

```python
from NeuralNet import NeuralNetwork, Activation

# 1. Define Architecture: Input(2) -> Hidden(4) -> Output(1)
nn = NeuralNetwork(structure=[2, 4, 1], activation=Activation.Sigmoid)
# or, if different activations per layer is required:
# nn = NeuralNetwork(structure=[2, 4, 1], activation=[Activation.Sigmoid, Activation.Relu])

# 2. Train
# X: Input data, y: Target labels
nn.train(X, y, epochs=1000, learning_rate=0.1)

# 3. Predict
predictions = nn.predict(X_test)
```

## 📂 Project Structure

```text
.
├── NeuralNet.py             # 🧠 The Core Class: Backprop & Forward pass logic
├── app_mnist.ipynb          # Implementation of the 784-16-16-10 network on MNIST
├── app.ipynb                # Flexible network playground
├── main.py                  # Script for testing structure
└── data/                    # Dataset storage
```

## 🤝 Contributing
Feel free to open a PR if you want to add new activation functions or optimizers!