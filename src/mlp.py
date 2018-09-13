import numpy as np

input = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\intro ci\\mlp-261456-2018\\data\\flood-input-normalized.csv', delimiter=',')
input_us = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\intro ci\\mlp-261456-2018\\data\\flood-input.csv', delimiter=',')

d = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\intro ci\\mlp-261456-2018\\data\\flood-d-normalized.csv', delimiter=',')
d_us = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\intro ci\\mlp-261456-2018\\data\\flood-desired-output.csv', delimiter=',')
d.shape = (len(d), 1)
d_us.shape = (len(d), 1)

# np.random.seed(1)

def activation(x, f, d=False):
    if(f == 'logistic'):
        if(d):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    
    elif(f == 'relu'):
        if(d):
            return 1. * (x > 0)
        return np.maximum(x, 0)
    
    else:
        print('No such ' + f + ' function implemented in this program.')
        exit()

def extend_bias(x):
    bias = np.ones((x.shape[0], 1))
    return np.concatenate((x, bias), axis=1)

def forward(layers, input=[]):
    if(len(input) > 0):
        layers[0].set_input(input)
    outputs = [layers[0].forward()]
    for l in layers[1:]:
        x = outputs[-1]
        a = l.forward(x)
        outputs.append(a)
    return outputs

def back(i, d, layers, outputs, lr):
    e = d
    means = []
    variances = []
    for o in reversed(outputs[1:]):
        m = np.zeros(o.shape)
        v = np.zeros(o.shape)
        means.append(m)
        variances.append(v)

    j = 0
    for l in reversed(layers[1:]):
        output = outputs.pop()
        input = outputs[-1]
        result = l.back(e, input, output, i, means[j], variances[j], lr)
        e = result[0]
        gradient = result[1]
        j += 1
    
def print_weights(layers):
    for l in layers[1:]:
        print(l.weight)

class Input(object):
    def __init__(self, input):
        self.x = input
    
    def forward(self):
        return self.x
    
    def set_input(self, input):
        self.x = input

class Hidden(object):
    def __init__(self, act, n_input, n):
        self.act = act
        self.weight = np.random.randn(n_input + 1, n) / np.sqrt(n_input) # xavier initialization

    def forward(self, input):
        input = extend_bias(input)
        z = input.dot(self.weight)
        a = activation(z, self.act)
        return a

    def back(self, e, input, output, t, m, v, lr, b1=.9, b2=.999, eps=1e-7):
        input = extend_bias(input)
        gradient = e * activation(output, self.act, d=True)

        m = b1 * m + (1 - b1) * gradient # use adam optimization in gradient descent
        v = b2 * v + (1 - b2) * (gradient ** 2)
        m_hat = m/(1 - np.power(b1, t))
        v_hat = v/(1 - np.power(b2, t))
        inertia = m_hat/(np.sqrt(v_hat) + eps)
        self.weight += (lr * np.amax(inertia)) * input.T.dot(gradient) # l0T dot grad
        return [gradient.dot(self.weight[:-1].T), gradient] # error of previous layer

class Output(object):
    def __init__(self, act, n_input, n):
        self.act = act
        self.weight = np.random.randn(n_input + 1, n) / np.sqrt(n_input) # xavier initialization

    def forward(self, input):
        input = extend_bias(input)
        z = input.dot(self.weight)
        a = activation(z, self.act)
        return a

    def back(self, d, input, output, t, m, v, lr, b1=.9, b2=.999, eps=1e-7):
        e = d - output
        input = extend_bias(input)
        
        gradient = e * activation(output, self.act, d=True)
        m = b1 * m + (1 - b1) * gradient # use adam optimization in gradient descent
        v = b2 * v + (1 - b2) * (gradient ** 2)
        m_hat = m/(1 - np.power(b1, t))
        v_hat = v/(1 - np.power(b2, t))
        inertia = m_hat/(np.sqrt(v_hat) + eps)
        self.weight += (lr * np.amax(inertia)) * input.T.dot(gradient) # l0T dot grad
        return [gradient.dot(self.weight[:-1].T), gradient] # error of previous layer

class MLP(object):
    def __init__(self, activations, form, input, output):
        self.acts = activations
        self.d = output
        self.form = form
        self.x = input
    
    def init_network(self, activations, form, activation=[]):
        layers = [Input(self.x)]
        for l in range(1, len(form)-1):
            layers.append(Hidden(activations[l], form[l-1], form[l]))
        layers.append(Output(activations[-1], form[-2], form[-1]))
        return layers

def k_fold(activations, form, input, d, k, learning_rate):
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

    sum_acc = 0
    models = []
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
        layers = train(activations, form, training_set[i], d_train[i], learning_rate)
        fold_acc = test(layers, testing_set[i], d_test[i])
        sum_acc += fold_acc
        models.append(layers)
        print('fold[' + str(i) + '] accuracy:', fold_acc)
    print('avg accuracy:', sum_acc / k)
    return models

def scale_back(s, a):
    return np.add(s * (np.amax(a) - np.amin(a)), np.amin(a))

def test(layers, testing_set, d_test):
    output = forward(layers, testing_set)[-1]
    out_us = scale_back(output, d_us)
    d_test_us = scale_back(d_test, d_us)
    compare = np.concatenate((out_us, d_test_us), axis=1)
    print('       y           d')
    print(compare)
    accuracy = 100 - ((np.abs(out_us - d_test_us) / d_test_us) * 100)
    avg_acc = np.mean(accuracy)
    return avg_acc

def train(activations, form, training_set, d_train, learning_rate):
    model = MLP(activations, form, training_set, d_train)
    layers = model.init_network(activations, model.form)
    cost = np.inf

    i = 1
    epochs = 2 ** 17
    while(cost > 1e-4 and i <= epochs):
        outputs = forward(layers)
        e = model.d - outputs[-1]
        cost = np.mean(np.abs(e))
        if(i % 32768 == 0):
            print(i, 'mse:' , cost)
        back(i, model.d, layers, outputs, learning_rate)
        i += 1
    return layers

learning_rate = .01
activations = ['logistic', 'logistic', 'logistic',]
form = [input.shape[1], 3, 3, 1]
k_fold(activations, form, input, d, 10, learning_rate)
