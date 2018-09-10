import numpy as np

input = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\intro ci\\mlp-261456-2018\\data\\flood-input-normalized.csv', delimiter=',')

d = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\intro ci\\mlp-261456-2018\\data\\flood-d-normalized.csv', delimiter=',')
d.shape = (len(d), 1)
    
x = np.array([[0.,0.],
            [0.,1.],
            [1.,0.],
            [1.,1.]])
                
y = np.array([[0.],
			[1.],
			[1.],
			[0.]])

np.random.seed(1)

def activation(x, d=False, f='logistic'):
    if(f == 'logistic'):
        if(d):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

def extend_bias(x):
    bias = np.ones((x.shape[0], 1))
    return np.concatenate((x, bias), axis=1)

class Input(object):
    def __init__(self, input):
        self.x = input
    
    def forward(self):
        return self.x

class Hidden(object):
    def __init__(self, n_input, n):
        self.weight = np.random.random((n_input, n))

    def forward(self, input):
        z = input.dot(self.weight)
        a = activation(z)
        return a

    def back(self, e, input, output):
        gradient = e * activation(output, d=True)
        self.weight += input.T.dot(gradient) # l0T dot grad
        return gradient.dot(self.weight.T)

class Output(object):
    def __init__(self, n_input, n):
        self.weight = np.random.random((n_input, n))

    def forward(self, input):
        z = input.dot(self.weight)
        a = activation(z)
        return a

    def back(self, d, input, output):
        e = d - output
        gradient = e * activation(output, d=True)
        self.weight += input.T.dot(gradient) # l0T dot grad
        return gradient.dot(self.weight.T) # error

class MLP(object):
    def __init__(self, form, input, output):
        self.d = output
        self.form = form
        self.x = input
    
    def init_network(self, form, activation=[]):
        layers = [Input(self.x)]
        for l in range(1, len(form)-1):
            layers.append(Hidden(form[l-1], form[l]))
        layers.append(Output(form[-2], form[-1]))
        return layers

def print_weights(layers):
    for l in layers[1:]:
        print(l.weight)

def forward(layers):
    outputs = [layers[0].forward()]
    for l in layers[1:]:
        x = outputs[-1]
        a = l.forward(x)
        outputs.append(a)
    return outputs

def back(d, layers, outputs):
    e = d
    for l in reversed(layers[1:]):
        output = outputs.pop()
        input = outputs[-1]
        e = l.back(e, input, output)


nn = MLP([x.shape[1], 3, 1], x, y)
activations = ['LOGISTIC', 'LOGISTIC']
layers = nn.init_network(nn.form)

cost = np.inf
i = 0
while(cost > 1e-3):
    outputs = forward(layers)
    e = nn.d - outputs[-1]
    cost = np.mean(np.abs(e))
    if(i % 5000 == 0):
        print(i, 'mse:' , cost)
    back(nn.d, layers, outputs)
    i += 1

# Prototype
# h = Hidden(2, 3)
# weight_o = 2 * np.random.random((3, 1)) -1
# for i in range(120000):
#     yh = h.forward(x)
#     # print('w-before', h.weight)
#     yo = activation(yh.dot(weight_o))
#     eo = y - yo
#     if(i % 5000 == 0):
#         print('error:' ,np.mean(np.abs(eo)))
#     ograd = eo * activation(yo, d=True)
#     weight_o += yh.T.dot(ograd)
#     eh = ograd.dot(weight_o.T)
#     h.back(eh, x, yh)
#     # print('w-after', h.weight)

# Forward Test
# input = np.asarray([[0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]])

# w0 = np.asarray([[7.48836465, 6.54532016, -5.32719126],
#  [-5.32126606 ,6.54034775 ,7.50611517]])
# w1 = np.asarray([[-19.68665978],
#  [ 26.67530248],
#  [-19.68571172]])

# a0 = activation(input.dot(w0))
# a1 = activation(a0.dot(w1))
# print(a1)
