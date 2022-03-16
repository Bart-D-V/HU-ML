from Neuron import Neuron
from typing import List


class NeuronLayer:
    """a layer has one or more neurons"""
    neurons: List[Neuron]

    def __init__(self, neurons: List[Neuron]):
        self.neurons = neurons
