from NeuronLayer import NeuronLayer
from typing import List


class NeuronNetwork:
    """a network has one or more layers"""
    neuron_layers: List[NeuronLayer]

    def __init__(self, neuron_layers: List[NeuronLayer]):
        self.neuron_layers = neuron_layers

    def feed_forward(self, inputs: List[float]) -> List[List[float]]:
        """gives ouput of a network given a input"""
        output_and_input = [[inputs]]
        """input and output is used to store the output of a layer and uses it as a input for the next layer"""
        for layer in self.neuron_layers:
            """loop through all the layers"""
            output_and_input.append(list(n.get_output(output_and_input[1][-1]) for n in layer.neurons))

        return output_and_input

    def backpropagation(self, targets: List[float], learning_rate: float):
        """calculates errors of the neurons.
        begins with the output-layer and calculates back to the input-layer.
        afterwards the weights adn bias get updated."""
        output_layer = self.neuron_layers[-1]

        for i in range(len(output_layer.neurons)):  # calculate error of output-layer neurons
            output_layer.neurons[i].output_error(targets[i])

        for layer in self.neuron_layers[-2::-1]:  # calculate error of hiddenlayer neurons from back to front
            for i in range(len(layer.neurons)):
                layer.neurons[i].hidden_output_error(output_layer.neurons, i)
            output_layer = layer

        for layer in self.neuron_layers:  # update the weights and bias
            for neuron in layer.neurons:
                neuron.update()

    def train(self, inputs: List[List[float]], targets: List[List[float]], epochs: int, learning_rate: float):
        """"uses the feed_forward and backpropagation functions to train a neuralnetwork on a inputs and target dataset"""
        for epoch in range(epochs):
            print(f"starting epoch {epoch}.\n",)
            for inputs, target in zip(inputs, targets):
                self.feed_forward(inputs)
                self.backpropagation(target, learning_rate)
        print(f"finished training, for all the {epochs} epochs!")