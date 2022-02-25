def print_neuron(neuron):
    """loop through all possible inputs and outputs"""
    for inp in all_possible_inputs(len(neuron.weights)):
        print(f"<met input: {inp} is output: {round(neuron.output(inp), 2)}>")  # output prints with 3 digits


def print_neuron_network(neuron_network):
    """loop through all possible inputs and outputs"""
    for inp in all_possible_inputs(len(neuron_network.neuron_layers)):
        outputs = neuron_network.feed_forward(inp)

        for i in range(len(outputs)):
            """round all outputs"""
            outputs[i] = round(outputs[i], 2)

        print(f"<met input: {inp} is de output: {outputs}>")  # output prints with 3 digits


def all_possible_inputs(a_inputs):
    inputs = []
    """make all possible inputs for a neuron or a network"""
    for i in range(1 << a_inputs):
        s = bin(i)[2:]
        s = '0' * (a_inputs - len(s)) + s
        inputs.append(list(map(int, list(s))))

    return inputs
