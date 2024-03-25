from test import test
import torch

def sigmoid(input):
    x = torch.exp(input)
    return x / (1+x)

data = torch.randn([4,2])
tester = test(torch.sigmoid,sigmoid)
tester.print_test(data)


def tanh(input):
    return 2*sigmoid(input) - 1

data = torch.randn([4,2])
tester = test(torch.tanh,tanh)
tester.print_test(data)


def relu(input):
    return torch.max(0,input)

data = torch.randn([4,4,4])
tester = test(torch.nn.functional.relu,relu)
tester.print_test(data)