import numpy as np
import sys

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
        j += 1
    
def print_weights(model):
    for l in model[0][1:]:
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

    def back(self, e, input, output, t, m, v, lr, b1=.9, b2=.98, eps=1e-5):
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

    def back(self, d, input, output, t, m, v, lr, b1=.9, b2=.98, eps=1e-5):
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

def k_fold(activations, form, input, d, k, learning_rate, epochs):
    # features[i] and label[i] must be at the same position after shuffling so concatenate them first
    data = np.concatenate((input, d), axis=1)
    np.random.shuffle(data)

    # this block separate features from output label
    input = np.hsplit(data, [input.shape[1]])[0]
    d = np.hsplit(data, [input.shape[1]])[1].reshape(d.shape) # data.T[-(d.shape[1]):].reshape(data.shape[0], d.shape[1])
    # partition into folds
    fold_len = int(data.shape[0] / k)
    input_folds = []
    output_folds = []
    for i in range(k):
        input_folds += [input[i * fold_len:(i+1) * fold_len]]
        output_folds += [d[i * fold_len:(i+1) * fold_len]]
    if(input.shape[0] % k > 0): # prevent empty array
        input_folds += [input[k * fold_len:input.shape[0]]]
        output_folds += [d[k * fold_len:d.shape[0]]]
    
    sum_acc = 0
    model = []
    for i in range(k):
        print('fold:', i)
        input_temp = input_folds.copy()
        output_temp = output_folds.copy()
        testing_set = input_temp[i]
        d_test = output_temp[i]
        del(input_temp[i])
        del(output_temp[i])
        training_set = np.concatenate(input_temp, axis=0)
        d_train = np.concatenate(output_temp, axis=0)
        layers = train(activations, form, training_set, d_train, learning_rate, epochs)
        fold_acc = test(layers, testing_set, d_test)
        sum_acc += fold_acc
        print('fold[' + str(i) + '] accuracy:', fold_acc, '%', '\n', '\n')
    model.append(layers)
    print('avg accuracy:', sum_acc / k, '%')
    return model

def scale_back(s, a):
    return np.add(s * (np.amax(a) - np.amin(a)), np.amin(a))

def test(layers, testing_set, d_test):
    output = forward(layers, testing_set)[-1]
    out_us = scale_back(output, d_us)
    d_test_us = scale_back(d_test, d_us)
    compare = np.concatenate((out_us, d_test_us), axis=1)
    print(' [          [y]        ][          [d]       ]')
    print(compare, '\n')
    
    if(d.shape[1] < 2): # prediction accuracy
        accuracy = 100 - ((np.abs(out_us - d_test_us) / (d_test_us + 1e-5)) * 100) # adding small value to prevent divided by 0
        avg_acc = np.mean(accuracy) # average predicton accuracy
        return avg_acc
    else: # this condition use only for cross.pat
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
        for i in range(d_test.shape[0]):
            # transform (y to [0, 1] or [1, 0])
            if(output[i][0] > output[i][1]):
                output[i] = np.array([1, 0])
            elif(output[i][0] < output[i][1]):
                output[i] = np.array([0, 1])
            # check (y, d) conditions
            if(np.array_equal(output[i], d_test[i])):
                if(np.array_equal(d_test[i], [0, 1])):
                    tp += 1
                elif(np.array_equal(d_test[i], [1, 0])):
                    tn += 1
                else:
                    if(np.array_equal(d_test[i], [0, 1])):
                        fp += 1
                    elif(np.array_equal(d_test[i], [1, 0])):
                        fn += 1
        print('confusion_matrix')
        print('    |', '[ y ]')
        print('[d] |', np.array([0, 1]))
        print('-----------')
        print(np.array([0]), '|', np.array([tp, tn]))
        print(np.array([1]), '|', np.array([fp, fn]), '\n')
        accuracy = ((tp + fn)/(tp + fp + tn + fn)) * 100
    return accuracy
    
def train(activations, form, training_set, d_train, learning_rate, epochs):
    model = MLP(activations, form, training_set, d_train)
    layers = model.init_network(activations, model.form)
    cost = np.inf

    i = 1
    while(cost > 1e-4 and i <= epochs):
        outputs = forward(layers)
        e = model.d - outputs[-1]
        cost = np.mean(np.abs(e))
        if(i % 32768 == 0):
            print(i, 'mse:' , cost)
        back(i, model.d, layers, outputs, learning_rate)
        i += 1
    return layers

if(__name__ == '__main__'):
    f = sys.argv[0]
    dataset = int(sys.argv[1])
    if(dataset == 2 or dataset == 'cross'):
        input = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\ci\\mlp-261456-2018\\data\\cross-input.csv', delimiter=',')
        input_us = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\ci\\mlp-261456-2018\\data\\cross-input.csv', delimiter=',')
        d = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\ci\\mlp-261456-2018\\data\\cross-output.csv', delimiter=',')
        d_us = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\ci\\mlp-261456-2018\\data\\cross-output.csv', delimiter=',')
    else:
        input = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\ci\\mlp-261456-2018\\data\\flood-input-normalized.csv', delimiter=',')
        input_us = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\ci\\mlp-261456-2018\\data\\flood-input.csv', delimiter=',')
        d = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\ci\\mlp-261456-2018\\data\\flood-d-normalized.csv', delimiter=',')
        d_us = np.genfromtxt('C:\\Users\\Patdanai\\Desktop\\ci\\mlp-261456-2018\\data\\flood-desired-output.csv', delimiter=',')

    try:
        d.shape = (d.shape[0], d.shape[1])
        d_us.shape = (d.shape[0], d.shape[1])
    except:
        d.shape = (d.shape[0], 1)
        d_us.shape = (d.shape[0], 1)
        pass

    form = sys.argv[2].split('-')
    activations = sys.argv[3].split('-')
    learning_rate = float(sys.argv[4])
    k = int(sys.argv[5])
    epochs = int(sys.argv[6])
    arguments = [f, dataset, form, activations, learning_rate, k, epochs]

    # define mlp parameters
    form = [input.shape[1]]
    [form.append(int(l)) for l in arguments[2]]
    form.append(d.shape[1])
    print(form)
    activations = arguments[3]
    model = k_fold(activations, form, input, d, k, learning_rate, epochs)
    print_weights(model)