import numpy as np

class MLP(object):
    def __init__(self, form, input, act_type): # [10(input(0th)), 10(hidden(1st)), 5(hidden(2nd)), 5(hidden(3rd)), 2(output(4th))]
        self.act_type = act_type
        self.form = form
        self.layer0 = Input(input)
        self.layers = [self.layer0]

    def init_network(self):
        act_type = self.act_type
        n = self.form
        [self.layers.append(Hidden(n[l-1], n[l], act_type[l-1])) for l in range(1, len(n)-1)]
        self.layers.append(Output(n[-2], n[-1], act_type[-1]))

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
    def __init__(self, input):
        self.input = input
        self.weight = 2 * np.random.random((len(self.input), 1)) - 1
    
    def forwarding(self):
        return np.dot(self.input, self.weight)

    def back_propagation(self, input):
        pass

class Hidden(object):
    def __init__(self, input, n, act_type):
        self.activation = Regression()
        self.act_type = act_type
        self.input = input
        self.weight = 2 * np.random.random((input, n)) -1
        
    def activation_fn(self, x, d=False):
        if(self.act_type == Regression.LOGISTIC):
            return self.activation.logistic(x)
        elif(self.act_type == Regression.TANH):
            return self.activation.tanh(x)

    def forwarding(self):
        return self.activation_fn(self.act_type, self.input)

    def back_propagation(self, grad):
        pass

class Output(object):
    def __init__(self, input, n, act_type):
        self.activation = Regression()
        self.act_type = act_type
    
    def activation_fn(self, x, d=False):
        if(self.act_type == Regression.LOGISTIC):
            return self.activation.logistic(x)
        elif(self.act_type == Regression.TANH):
            return self.activation.tanh(x)

    def forwarding(self):
        return 1
    
    def back_propagation(self, output, grad):
        output = self.forwarding()
        return self.activation_fn(output, d=True) * grad

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
act = [reg.LOGISTIC, reg.LOGISTIC, reg.TANH]
mlp = MLP(f, a, act)
layers = mlp.init_network()
print(layers)
print(layers[0].input)
print(layers[1].input)