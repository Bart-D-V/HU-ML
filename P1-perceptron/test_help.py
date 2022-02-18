def print_perceptron(perceptron):
    """loop through all possible inputs and outsputs"""
    for inp in all_possible_inputs(len(perceptron.weights)):
        print("met input: " + str(inp) + " is output: " + str(perceptron.activation(inp)))


def print_perceptron_network(perceptron_network):
    """loop through all possible inputs and outsputs"""
    for inp in all_possible_inputs(len(perceptron_network.perceptron_layers)):
        print("met input: " + str(inp) + " is de output: " + str(perceptron_network.feed_forward(inp)))


def all_possible_inputs(a_inputs):
    inputs = []
    """make all possible inputs for perceptron or network"""
    for i in range(1 << a_inputs):
        s = bin(i)[2:]
        s = '0' * (a_inputs - len(s)) + s
        inputs.append(list(map(int, list(s))))

    return inputs
