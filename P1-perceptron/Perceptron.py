from PerceptronLayer import PerceptronLayer


class Perceptron(PerceptronLayer):

    def __init__(self, weights, inputs, bias):
        self.weights = weights
        self.inputs = inputs
        self.bias = bias

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def get_input(self):
        return self.inputs

    def set_weights(self, weights):
        self.weights = weights

    def set_bias(self, bias):
        self.bias = bias

    def set_inputs(self, inputs):
        self.inputs = inputs

    def activation(self):
        input_sum = 0
        for i in range(len(self.weights)):
            input_sum += self.weights[i] * self.inputs[i]
        if input_sum + self.bias >= 0:
            return 1
        else:
            return 0

    def __str__(self):
        return "Met weights: " + str(self.weights) + ", input: " + str(self.inputs) + " en bias: " + str(self.bias) + "\nis de output: " + str(
            self.activation())


"""tests"""
#INVERT
print("INVERT-poort")
INVERT = Perceptron([-1], [1], 0)
print(INVERT.__str__())
INVERT.set_inputs([0])
print(INVERT.__str__())

#AND
print("\nAND-poort")
AND = Perceptron([0.5, 0.5], [1, 1], -1)
print(AND.__str__())
AND.set_inputs([0, 1])
print(AND.__str__())
AND.set_inputs([0, 0])
print(AND.__str__())

#OR
print("\nOR-poort")
OR = Perceptron([0.5, 0.5], [1, 1], -0.5)
print(OR.__str__())
OR.set_inputs([0, 1])
print(OR.__str__())
OR.set_inputs([0, 0])
print(OR.__str__())

#NOR
print("\nNOR-poort")
NOR = Perceptron([-1, -1, -1], [1, 1, 1], 0)
print(NOR.__str__())
NOR.set_inputs([1, 0, 1])
print(NOR.__str__())
NOR.set_inputs([0, 0, 0])
print(NOR.__str__())

#figuur 2.8 uit de reader
print("\nfiguur 2.8 uit de reader")
rp = Perceptron([0.6, 0.3, 0.2], [1, 1, 1], -0.4)
print(rp.__str__())
rp.set_inputs([0, 1, 1])
print(rp.__str__())
rp.set_inputs([1, 0, 0])
print(rp.__str__())
rp.set_inputs([0, 0, 1])
print(rp.__str__())
