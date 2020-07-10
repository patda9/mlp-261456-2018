import numpy as np

from enum import Enum

class Activation(Enum):
    ELU = "elu"
    LEAKY_RELU = "lrelu"
    LOGISTIC = "logistic"
    RELU = "relu"
    SIGMOID = "logistic"
    SOFTMAX = "softmax"
    TANH = "tanh"

class ActivationFunctions:
    def __init__(self, activation, _input, derivative = False):
        self.activation = activation
        self.derivative = derivative
        self.x = _input

        self.value = self.calculate(self.activation)

    def calculate(self, activation):
        if (activation == Activation.LOGISTIC):
            return self.logistic(self.x, self.derivative)
        elif (activation == Activation.RELU):
            return self.relu(self.x, self.derivative)
        elif (activation == Activation.SIGMOID):
            return self.logistic(self.x, self.derivative)
        # elif (activation == Activation.SOFTMAX):
        #     return self.softmax(self.x, self.derivative)
        elif (activation == Activation.TANH):
            return self.tanh(self.x, self.derivative)
    # def elu(self, x):
    #     pass

    # def lrelu(self, x):
    #     pass

    def logistic(self, x, derivative = False):
        f = lambda x: 1. / (1. + np.exp(-x))

        return f(x) * (1. - f(x)) if (derivative) else f(x)

    def relu(self, x, derivative = False):
        f = lambda x: np.max([0., x], 0)

        return (1. if x > 0. else 0.) if (derivative) else f(x)
    
    # def softmax(self, x, derivative = False):
    #     _sum = np.sum(np.exp(x))
    #     f = lambda x: np.exp(x) / _sum
    #     return f(x)

    def tanh(self, x, derivative = False):
        f = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        return 1. - (f(x) ** 2) if (derivative) else f(x)

if __name__ == "__main__":
    activation = ActivationFunctions(Activation.RELU, np.random.randn())

    print(activation.value)