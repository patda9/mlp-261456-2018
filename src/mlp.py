import numpy as np

def write_to_file(s_array, l):
    with open('test' + str(l) + '.txt', 'w') as f:
        for el in s_array:
            f.write(str(el) + str('\n'))

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
        self.layers = [self.layer0]  # initialize l=0 as input layer
        [self.layers.append(Hidden(self.layers[i-1].forwarding(), n[i+1], act_type[i-1])) for i in range(1, len(n)-1)]  # initialize 0<l<len(n)-1 as hidden layer
        self.layers.append(Output(self.layers[-1].forwarding(), self.d, act_type[-1])) # initialize l=len(n) layer as output layer
        return self.layers

    def run(self):
        # init_network -> back_prop -> forward : -1 epoch from forward initialization
        pass

    def forwarding(self):  # TODO ###
        # input = self.layers[0]
        # outputs = [input.forwarding()]
        l = 0
        for layer in self.layers:
            # x = outputs[-1]
            y = layer.forwarding()
            write_to_file(y, l)
            l += 1
            # outputs.append(y)
        # return outputs
    def calculate_loss(self):
        pass

    def back_propagation(self):
        pass

    def train(self, k):
        # 10-fold (10%) cross validation
        pass

    def test(self):
        pass


class Input(object):
    def __init__(self, input, n):
        self.x = input
        self.weight = np.random.random((len(self.x[0])+1, n)) # add 1 dimension for extended bias

    def extend_bias(self, x):
        bias = np.ones((len(self.x), 1))
        return np.concatenate((x, bias), axis=1)

    def forwarding(self):
        x = self.x
        x = self.extend_bias(x)
        self.z = np.dot(x, self.weight)
        return self.z

    def back_propagation(self, input, grad, learning_rate):
        # input = extend_input_bias(input).T
        delta_w = input.dot(grad)
        self.weight += -learning_rate * delta_w
        return grad.dot(self.weight[:-1].T)

class Hidden(object):
    def __init__(self, input, n, act_type):
        self.activation = Regression()
        self.act_type = act_type
        self.x = input
        self.weight = np.random.random((len(self.x[0])+1, n))
    
    def extend_bias(self, x):
        bias = np.ones((len(self.x), 1))
        return np.concatenate((x, bias), axis=1)
    
    def hidden_net(self):
        x = self.x
        x = self.extend_bias(x)
        self.z = np.dot(x, self.weight)
        return self.z

    def activation_fn(self, z, d=False):
        if(self.act_type == Regression.LOGISTIC):
            self.a = self.activation.logistic(z)
            return self.a
        elif(self.act_type == Regression.TANH):
            self.a = self.activation.tanh(z)
            return self.a
        elif(self.act_type == Regression.RELU):
            self.a = self.activation.relu(z)
            return self.a

    def forwarding(self):
        z = self.hidden_net()
        a = self.activation_fn(z)
        return a

    def back_propagation(self, grad, learning_rate):
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
        return self.e * self.activation_fn(self.y, d=True)
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
d.shape = (len(d), 1)

nn = MLP([len(input[0]), 5, 5, 1], input, d, [reg.RELU, reg.RELU, reg.RELU])
nn.init_network()
# print(nn.layers[1].forwarding())
# print(len(nn.layers[1].x))
# print(nn.layers[2].forwarding())
# print(nn.layers[-1])
# print(nn.layers[1].a)
# print(nn.layers[1].hidden_net())
print(nn.layers[1].a)
nn.forwarding()
print(nn.layers[-1].e)
print(nn.layers[2].extend_bias(nn.layers[2].x))
