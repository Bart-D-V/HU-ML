from __future__ import annotations
from typing import List
import math


def sigmoid(x: float):
    return 1 / (1 + math.e ** (-x))


def deriv_sigmoid(x: float):
    return sigmoid(x) * (1 - sigmoid(x))


class Neuron:
    """parameters neuron"""
    weights: List[float]
    bias: float
    error: float
    inputs: List[float]
    input_weight_sum: float
    output: float
    weight_deltas: List[float]
    bias_delta: float

    def __init__(self, weights: List[float], bias: float):
        """give a list of weights and a bias for neuron"""
        self.weights = weights
        self.bias = bias

    def get_output(self, inputs: List[float]):
        """give output of a neuron with a given input, weights and bias.
        also saves the input, input sum and the output to use later to calculate the deltas"""
        self.inputs = inputs
        self.input_weight_sum = sum([input_value * weight for input_value, weight in zip(inputs, self.weights)]) + self.bias
        self.output = sigmoid(self.input_weight_sum)
        return self.output

    def output_error(self, target: float):
        """calculates the error of the output of a outputneuron"""
        self.error = deriv_sigmoid(self.input_weight_sum) * -(target - self.output)
        return self.error

    def hidden_output_error(self, neurons: List[Neuron], layer_index: int):
        """calculates the error of the output of a hiddenneuron"""
        self.error = sum([n.error * n.weights[layer_index] for n in neurons]) * deriv_sigmoid(self.input_weight_sum)
        return self.error

    def delta_weight(self, learning_rate, output):
        """calculate delta of a weight"""
        return learning_rate * output * self.error

    def deltas_weights_bias(self, learning_rate):
        """calculates the delta bias and saves the deltas of the weights and bias to update later when all the deltas
        are calculated """
        deltas = []
        for i in range(len(self.weights)):
            deltas.append(self.delta_weight(learning_rate, self.inputs[i]))

        self.weight_deltas = deltas
        self.bias_delta = learning_rate * self.error

    def update(self):
        """update weights and bias with deltas"""
        self.weights = [weight - delta for weight, delta in zip(self.weights, self.weight_deltas)]
        self.bias = self.bias - self.bias_delta

    def __str__(self):
        return f"<Neuron weights: {self.weights} en bias: {self.bias}>"
