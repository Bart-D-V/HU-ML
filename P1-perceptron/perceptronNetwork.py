from perceptronLayer import PerceptronLayer


class PerceptronNetwork:

    perceptron_layers: list[PerceptronLayer]

    def __init__(self, perceptron_layers: list[PerceptronLayer]):
        self.perceptron_layers = perceptron_layers
