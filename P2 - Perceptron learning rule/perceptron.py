from typing import List


def loss(preds: List[float], targs: List[float]):
    """calculates the total mean squared error of all the training iterations"""
    return sum([(t - p) ** 2 for t, p in zip(targs, preds)]) / len(targs)


class Perceptron:
    """parameters perceptron"""
    weights: List[float]
    bias: float
    learning_rate = 0.1

    def __init__(self, weights: List[float], bias: float):
        """give a list of weights and a bias for perceptron"""
        self.weights = weights
        self.bias = bias

    def activation(self, inputs: List[float]):
        """ activation of the weights if input is 1. then give the sum of the weights + bias"""
        weight_sum = sum([input_value * weight for input_value, weight in zip(inputs, self.weights)]) + self.bias
        """check if sum of weights > threshold"""
        if weight_sum >= 0:
            return 1
        else:
            return 0

    def update(self, inputs: List[float], target: float):
        """update the perceptron weights and bias by means of the perceptron learning rule -
         weights/bias += learning_rate * error """
        pred = self.activation(inputs)
        error = target - pred
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * inputs[i]
        self.bias += self.learning_rate * error

    def __str__(self):
        return f"<Perceptron weights: {self.weights} en bias: {self.bias}>"
