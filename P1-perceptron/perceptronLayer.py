from perceptron import Perceptron


class PerceptronLayer:

    perceptrons: list[Perceptron]

    def __init__(self, perceptrons: list[Perceptron]):
        self.perceptrons = perceptrons
