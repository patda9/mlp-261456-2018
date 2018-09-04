import numpy as np

class MLP(object):
    def __init__(self, form, input, output, act_type): # [10(input(0th)), 10(hidden(1st)), 5(hidden(2nd)), 5(hidden(3rd)), 2(output(4th))]
        self.act_type = act_type
        self.form = form
        self.layer0 = Input(input)
        self.layers = [self.layer0]

    def init_network(self):
        act_type = self.act_type
        n = self.form
        [self.layers.append(Hidden(n[l-1], n[l], act_type[l-1])) for l in range(1, len(n)-1)]
        # self.layers.append(Output(n[-2], n[-1], act_type[-1]))

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
        self.weight = 2 * np.random.random((len(self.x[0]), 5)) - 1
    
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
        self.weight = 2 * np.random.random((len(self.x[0]), n)) -1
        
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

    def forwarding(self):
        z = self.hidden_net()
        a = self.activation_fn(z)
        return a

    def back_propagation(self, grad):
        return

class Output(object):
    def __init__(self, input, d, n, act_type):
        self.activation = Regression()
        self.act_type = act_type
        self.x = input
        self.d = d
    
    def activation_fn(self, z, d=False):
        if(self.act_type == Regression.LOGISTIC):
            return self.activation.logistic(z)
        elif(self.act_type == Regression.TANH):
            return self.activation.tanh(z)

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
    TANH = 'TANH'

    def logistic(self, x, d=False):
        if(d):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x, d=False):
        if(d):
            return 1 - x ** 2
        return np.tanh(x)

reg = Regression()
a = np.random.random((6, 1))
f = [6, 5, 3, 3, 2]

input = np.genfromtxt('..\\data\\flood.csv', delimiter=',')
layer0 = Input(input, 5)
print(layer0.weight)
print('******', layer0.forwarding())
layer1 = Hidden(layer0.forwarding(), 2, reg.LOGISTIC)
print('******', layer1.x)
print(layer1.forwarding())