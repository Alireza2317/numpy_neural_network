from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np


class ActivationType(Enum):
	Sigmoid = "sigmoid"
	Relu = "relu"
	Tanh = "tanh"
	NoAct = "no-activation"


type VectorFunc = Callable[[np.ndarray], np.ndarray]


@dataclass
class ActivationPair:
	f: VectorFunc
	df: VectorFunc


def sigmoid(z: np.ndarray) -> np.ndarray:
	return 1 / (1 + np.exp(-z))


def d_sigmoid(z: np.ndarray) -> np.ndarray:
	return np.exp(-z) / (np.pow((1 + np.exp(-z)), 2))


def relu(z: np.ndarray) -> np.ndarray:
	return np.maximum(0, z)


def d_relu(z: np.ndarray) -> np.ndarray:
	return (z > 0).astype(np.float64)


def tanh(z: np.ndarray) -> np.ndarray:
	return np.tanh(z)


def d_tanh(z: np.ndarray) -> np.ndarray:
	return 4 * np.exp(2 * z) / np.power(np.exp(2 * z) + 1, 2)


def softmax(z: np.ndarray) -> np.ndarray:
	return np.exp(-z) / np.sum(np.exp(-z))
