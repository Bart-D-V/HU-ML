def all_possible_inputs(a_inputs):
    inputs = []
    for i in range(1 << a_inputs):
        s = bin(i)[2:]
        s = '0' * (a_inputs - len(s)) + s
        inputs.append(list(map(int, list(s))))

    return inputs
