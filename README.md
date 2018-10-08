# mlp-261456-2018
A Fully connected Multi-layer Perceptron using Python3

## Usage:
### Input parameters in command line for the multi-layer perceptron model
```
1. Input:
  1.1 Dataset (flood): input <int> 1
  1.2 Dataset (cross): input <int> 2
  
2. Network architecture: form <string>
* Example: 5 hidden nodes connected to second layer with 3 hidden nodes output nodes vary in size of 
desired output classes

- 5-3

3. Activation units: activations <string>
* Example: ReLu (Hidden1), Sigmoid(Hidden2), Sigmoid(Output)

- relu-logistic-logistic

4. Folds: k<int>; the number folds of folds used in k-fold cross validation

- 10

5. Learning rate in training: learning_rate <float>; where learning_rate is in [0, 1]

- 0.001

6. Training epochs epochs <int>

- 4096
```
### How to run the script
```
> cd ./src
> python mlp.py <dataset> <number of hidden layers> <activation_functions (include output node)>
 <k> <learning rate> <epochs>

Description: Flood dataset, 5-3-1 Neural network architecture, Sigmoid activation unit for all 
layers(Hidden1, Hidden2, Output), 10% cross validation with 0.001 learning rate and 4096 training 
epochs per fold

* Example > python mlp.py 1 5-3 logistic-logistic-logistic 10 0.001 4096
```
