import numpy as np

input = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\intro ci\\mlp-261456-2018\\data\\flood-input-normalized.csv', delimiter=',')

d = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\intro ci\\mlp-261456-2018\\data\\flood-d-normalized.csv', delimiter=',')
d.shape = (len(d), 1)

x = np.array([[0.,0.],
            [0.,1.],
            [1.,0.],
            [1.,1.],
            [0.,0.],
            [0.,1.],
            [1.,0.],
            [1.,1.],
            [0.,0.],
            [0.,1.],
            [1.,0.],
            [0.,0.],
            [0.,0.],
            [0.,1.],
            [1.,0.],
            [1.,1.],
            [0.,0.],
            [0.,1.],
            [1.,0.],
            [0.,0.]])
                
y = np.array([[0.],
			[1.],
			[1.],
			[0.],
            [0.],
			[1.],
			[1.],
			[0.],
            [0.],
			[1.],
			[1.],
			[1.],
            [0.],
			[1.],
			[1.],
			[0.],
            [0.],
			[1.],
			[1.],
			[1.]])

np.random.seed(1)

def activation(x, d=False, f='logistic'):
    if(f == 'logistic'):
        if(d):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    
    if(f == 'relu'):
        if(d):
            return 1. * (x > 0)
        return np.maximum(x, 0)

def extend_bias(x):
    bias = np.ones((x.shape[0], 1))
    return np.concatenate((x, bias), axis=1)

def forward(layers, input=None):
    if(input != None):
        outputs = [input.forward()]
    else:
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

def print_weights(layers):
    for l in layers[1:]:
        print(l.weight)

class Input(object):
    def __init__(self, input):
        self.x = input
    
    def forward(self):
        return self.x

class Hidden(object):
    def __init__(self, n_input, n):
        self.weight = np.random.randn(n_input + 1, n) / np.sqrt(n_input) # xavier initialization

    def forward(self, input):
        input = extend_bias(input)
        z = input.dot(self.weight)
        a = activation(z)
        return a

    def back(self, e, input, output):
        input = extend_bias(input)
        gradient = e * activation(output, d=True)
        self.weight += input.T.dot(gradient) # l0T dot grad
        return gradient.dot(self.weight[:-1].T) # error of previous layer

class Output(object):
    def __init__(self, n_input, n):
        self.weight = np.random.randn(n_input + 1, n) / np.sqrt(n_input) # xavier initialization

    def forward(self, input):
        input = extend_bias(input)
        z = input.dot(self.weight)
        a = activation(z)
        return a

    def back(self, d, input, output):
        e = d - output
        input = extend_bias(input)
        gradient = e * activation(output, d=True)
        self.weight += input.T.dot(gradient) # l0T dot grad
        return gradient.dot(self.weight[:-1].T) # error of previous layer

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

def k_fold(form, input, d, k):
    # features[i] and label[i] must be at the same position after shuffling so concatenate them first
    data = np.concatenate((input, d), axis=1)
    np.random.shuffle(data)
    
    # this block separate features from output label
    input = data[:, :-1]
    d = data.T[-1].reshape(data.shape[0], 1) 

    # partition into folds
    fold_len = int(data.shape[0]/k)
    input_folds = []
    output_folds = []
    for i in range(k):
        input_folds += [input[i * fold_len:(i+1) * fold_len]]
        output_folds += [d[i * fold_len:(i+1) * fold_len]]
    input_folds += [input[k * fold_len:input.shape[0]]]
    output_folds += [d[k * fold_len:d.shape[0]]]

    for i in range(k):
        print('fold:', i)
        input_temp = input_folds.copy()
        output_temp = output_folds.copy()
        testing_set = input_temp
        d_test = output_temp
        del(input_temp[i])
        del(output_temp[i])
        training_set = input_temp
        d_train = output_temp
        print(training_set[i].shape)
        print(d_train[i].shape)
        print(testing_set[i].shape)
        print(d_test[i].shape)
        test(train(form, training_set[i], d_train[i]), testing_set[i], d_test[i])

def test(layers, testing_set, d_test):
    output = forward(layers)[-1]
    out_temp = output.copy()
    d_test_temp = d_test.copy()
    compare = np.concatenate((out_temp, d_test_temp), axis=1)
    print(compare)
    print('test mse:', np.mean(np.abs(d_test - output)))

def train(form, training_set, d_train):
    model = MLP(form, training_set, d_train)
    layers = model.init_network(model.form)
    cost = np.inf
    i = 0
    epochs = 60000
    while(cost > 1e-3 and i < epochs):
        outputs = forward(layers)
        e = model.d - outputs[-1]
        cost = np.mean(np.abs(e))
        if(i % 5000 == 0):
            print(i, 'mse:' , cost)
        back(model.d, layers, outputs)
        i += 1
    return layers

form = [input.shape[1], 3, 3, 1]
k_fold(form, input, d, 10)


# Program section
nn = MLP([x.shape[1], 3, 3, 1], x, y)
activations = ['relu', 'relu', 'relu']
layers = nn.init_network(nn.form, activations)

# Training section
# cost = np.inf
# i = 0
# while(cost > 1e-3):
#     outputs = forward(layers)
#     e = nn.d - outputs[-1]
#     cost = np.mean(np.abs(e))
#     if(i % 5000 == 0):
#         print(i, 'mse:' , cost)
#     back(nn.d, layers, outputs)
#     i += 1

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
