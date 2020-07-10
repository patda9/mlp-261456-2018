import numpy as np

from activation_functions import Activation

class Node:
    def __init__(self,
                 activation = None,
                 bias = 1,
                 weight = None):
        self.activation = Activation.LOGISTIC if (activation == None) else activation
        self.weight = np.random.randn() if (weight == None) else weight
        self.bias = bias
