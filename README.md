# milligrad

This repo is a learning exercise for me to get a deeper understanding in deep learning and especially in dense neural networks.
Basically I took Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and added just some extra functions in order to train a neural network with just a bunch of lines of code.

## DenseNN initialization

To create a neural network in **milligrad** you have to call a **DenseNN** object. You have to specify the following parameters:

* *nin*: number of inputs of the first layer of the neural network. It is an integer and it has to match the size of ```inputs``` (a parameter of ```nn.Train()``` method).

* *nouts*: a list (or tuple) specifiying the number of neurons of each neural network layer. Note that it must be a 1-dimensional array and its length must match the one of ```activations``` array.

* *activations*: a list (or tuple) of strings specifiying the activations of each neural network layer. Note that it must be a 1-dimensional array and its length must match the one of ```nouts``` array. Right now only 5 activations are available: ```linear```, ```ReLU```, ```sigmoid```, ```tanh```, ```softmax```.

One useful **DenseNN** method is ```nn.parameters()``` with which you can access **DenseNN**'s parameters (weights and biases). Every parameter is a **Value** object containing 2 attributes:

* *data*: the numerical value of that neural network weight or bias.

* *grad*: the gradient that has been computed by backward propagation.

By printing a **DenseNN** you get a syntethic representation. It has to be read from left to right and it shows the structure of every layer of the neural network. For example:

    [Layer of [ReLUNeuron(3), ReLUNeuron(3)], Layer of [TanhNeuron(2)]]
means that the first layer is made of:

    2 neurons with ReLU activation, each one with 3 input neurons

and the second layer is made of:

    1 neuron with tanh activation and it takes 2 neurons (previous layer dimension) as inputs

## Training and Predictions

After initializing the neural network, essentially you have to use only 2 other functions: ```nn.Train()``` and ```nn.Predict()```:

* *nn.Train()* takes as inputs:
  * *inputs*: a list containing inputs value to feed in the ```nn``` neural network in order to train it. Note that inputs must be a 2-dimensional array (i.e. ```[[1, 2, 3, ...], [4, 5, 6, ...], [7, 8, 9, ...], ...]```).

  * *predictions*: a list containing predictions to feed in the ```nn``` neural network in order to train it. Note that inputs must be a 1-dimensional array if you want binary or linear classification (i.e. ```[1, 2, 3, ...]```) or a 2-dimensional array if you want a multi-class classification (i.e. ```[[1, 0, 0, ...], [0, 1, 0, ...], [0, 0, 1, ...], ...]```). Note that in the latter case ```predictions``` array must be one-hot encoded.

  * *loss*: a string containing the name of the loss function that has to be used in order to train the neural network. Currently available loss functions are: ```MeanSquaredError``` and ```BinaryCrossEntropy```, ```CategoricalCrossEntropy```. By default ```loss``` is set to ```''```. Since no loss function is applied by default, make sure to set properly this parameter.

  * *epochs_number*: an integer that specifies the number of training steps.
  
  * *batch_size*: an integer that specifies the batch size for gradient descent (hence how many examples the training function has to take before updating the weights and biases). For example ```batch_size=1``` corresponds to stochastic gradient descent, while ```batch_size=len(inputs)``` corresponds to batch gradient descent (every batch size between ```1``` and ```len(inputs)``` corresponds to mini batch gradient descent).
  
  * *learning_rate*: a float containing the learning rate for the weights update (param = param - learning_rate * gradient).

  * *decay_rate*: a float specifying the decay rate for learning rate decay (```learning_rate = learning_rate/(1 + decay_rate * epoch)```). By default is set to 0, so no learning rate decay is done without setting this parameter. Note that high ```learning_rate_decay``` makes learning rate decays faster.
  
  * *L2_regularization*: a float specifying the regularization factor for L2 regularization (loss = loss + L2_regularization * sum(weights^2)). Note that by default L2_regularization is set to 0 so no regularization is done.
* *nn.Train()* gives as output: a **DenseNN** object

* *nn.Predict()* takes as inputs:

  * *input*: a list containing one training set sample that will be fed in the neural network. Note that ```input``` length must match the size of ```nn``` zero-th layer.
* *nn.Predict()* gives as output the final node (a **Value** object) of the ```nn``` neural network containing the prediction value (```nn.Predict(input).data```).
