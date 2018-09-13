# mlp-261456-2018
A Fully connected Multi-layer Perceptron using Python3

## Usage:
### 1.How to run the script
```
> cd ./src
> python mlp.py
```
### 2.Input parameters for the multi-layer perceptron model
```
1. Input:
  1.1 Input (scaled): input = <path to .csv file>
  1.1 Input (unscaled): input_us = <path to .csv file>
  
2. Desired output:
  2.1 Desired output (sclaed): d = <path to .csv file>
  2.2 Desired output (unsclaed): d_us = <path to .csv file>
  
3. Network formation: form = <array>
* Example: 300 dimensions input with first layer with 5 hidden nodes connected to second layer with 3 hidden nodes and 2 output nodes
* form = [300, 5, 3, 2]

4. Activation units: activations = <array>
* Example: ReLu, Sigmoid, Sigmoid
* activations = ['relu', 'logistic', 'logistic']

5. Learning rate of training: learning_rate = <float> where learning_rate is in [0, 1]

6. Use k_fold(activations, form, input, d, k, learning_rate) as a main function where k = folds
```
