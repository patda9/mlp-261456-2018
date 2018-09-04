import numpy as np

class MLP(object):
    def __init__(self, form, input, output, act_type):
        self.act_type = act_type
        self.d = output
        self.form = form
        self.layer0 = Input(input, form[1])
        self.layers = [self.layer0]

    def init_network(self):
        act_type = self.act_type
        n = self.form
        self.layers = [layer0] # initialize l=0 as input layer
        [self.layers.append(Hidden(self.layers[i-1].forwarding(), n[i+1], act_type[i-1])) for i in range(1, len(n)-1)] # initialize 0<l<len(n)-1 as hidden layer
        self.layers.append(Output(self.layers[-1].forwarding(), self.d, act_type[-1])) #initialize l=len(n) layer as output layer
        return self.layers
    
    def run(self):
        pass
    
    def forwarding(self):
        input = self.layers[0]

        outputs = [input.forwarding()]
        for layer in self.layers[1:]:
            x = outputs[-1]
            y = layer.forward(x)
            outputs.append(y)

        return outputs

    def back_propagation(self):
        pass

    def train(self):
        pass
    
    def test(self):
        pass

class Input(object):
    def __init__(self, input, n):
        self.x = input
        self.weight = np.random.random((len(self.x[0]), 5))
    
    def forwarding(self):
        self.z = np.dot(self.x, self.weight)
        return self.z

    def back_propagation(self, input, gradient, learning_rate):
        # input = extend_input_bias(input).T
        delta_w = input.dot(gradient)
        self.weight += -learning_rate * delta_w
        return gradient.dot(self.weight[:-1].T)

class Hidden(object):
    def __init__(self, input, n, act_type):
        self.activation = Regression()
        self.act_type = act_type
        self.x = input
        self.weight = np.random.random((len(self.x[0]), n))
        
    def hidden_net(self):
        self.z = np.dot(self.x, self.weight)
        return self.z

    def activation_fn(self, z, d=False):
        if(self.act_type == Regression.LOGISTIC):
            self.a = self.activation.logistic(z)
            return self.a
        elif(self.act_type == Regression.TANH):
            self.a = self.activation.tanh(z)
            return self.a
        elif(self.act_type == Regression.RELU):
            return self.activation.relu(z)

    def forwarding(self):
        z = self.hidden_net()
        a = self.activation_fn(z)
        return a

    def back_propagation(self, grad):
        return

class Output(object):
    def __init__(self, input, d, act_type):
        self.activation = Regression()
        self.act_type = act_type
        self.x = input
        self.d = d
    
    def activation_fn(self, z, d=False):
        if(self.act_type == Regression.LOGISTIC):
            return self.activation.logistic(z)
        elif(self.act_type == Regression.TANH):
            return self.activation.tanh(z)
        elif(self.act_type == Regression.RELU):
            return self.activation.relu(z)

    def forwarding(self):
        self.y = self.activation_fn(self.x)
        self.e = self.d - self.y
        return self.y
    
    def back_propagation(self):
        return self.e * self.activation_fn(self.y)
        # output = self.forwarding()
        # return self.activation_fn(output, d=True) * grad

class Regression(object):
    LOGISTIC = 'LOGISTIC'
    RELU = 'RELU'
    TANH = 'TANH'

    def logistic(self, x, d=False):
        if(d):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x, d=False):
        if(d):
            return 1. * (x > 0)
        return np.maximum(x, 0)

    def tanh(self, x, d=False):
        if(d):
            return 1 - x ** 2
        return np.tanh(x)


""" 
program testing section 
"""

reg = Regression()
a = np.random.random((6, 1))
f = [6, 5, 3, 3, 2]

input = np.genfromtxt('..\\data\\flood-input.csv', delimiter=',')
d = np.genfromtxt('..\\data\\flood-desired-output.csv', delimiter=',')
layer0 = Input(input, 5)
print(layer0.weight)
print('******', layer0.forwarding())
layer1 = Hidden(layer0.forwarding(), 3, reg.RELU)
print('******', layer1.x)
print(layer1.forwarding())
layer2 = Hidden(layer1.forwarding(), 1, reg.RELU) # 1 is number of output nodes
print(layer2.forwarding())
layer3 = Output(layer2.forwarding(), d, reg.RELU)
print(layer3.forwarding())

nn = MLP([len(input[0]), 5, 5, 1], input, d, [reg.TANH, reg.RELU, reg.LOGISTIC])
nn.init_network()

print(nn.layers[-1])