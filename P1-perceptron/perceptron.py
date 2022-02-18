class Perceptron:
    weights: [float]
    bias: float

    def __init__(self, weights: list[float], bias: float):
        self.weights = weights
        self.bias = bias

    def activation(self, inputs: list[float]):
        input_sum = sum([input_value * weight for input_value, weight in zip(inputs, self.weights)]) + self.bias
        if input_sum + self.bias >= 0:
            return 1
        else:
            return 0

    def __str__(self):
        return "Perceptron weights: " + str(self.weights) + " en bias: " + str(self.bias)
