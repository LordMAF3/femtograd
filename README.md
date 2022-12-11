# femtograd

## DenseNN initialization

To create a neural network in **femtograd** you have to call a **DenseNN** object. You have to specify the following parameters:

* *nin*: number of inputs of the first layer of the neural network. It is an integer and it has to match the size of ```inputs``` (a parameter of ```nn.Train()``` method).

* *nouts*: a list (or tuple) specifiying the number of neurons of every neural network layer. Note that it must be a 1-dimensional array and its length must match the one of ```activations``` array. Moreover last layer has to be 1 since multi-class classification neural networks aren't supported yet.

* *activations*: a list (or tuple) of strings specifiying the activations of every neural network layer. Note that it must be a 1-dimensional array and its length must match the one of ```nouts``` array. Right now only 4 activations are available: ```'linear'```, ```'ReLU'```, ```'sigmoid'```, ```'tanh'```.

One useful **DenseNN** method is ```nn.parameters()``` (no parameters are to be specified) with which you can access **DenseNN**'s parameters (weights and biases). Every parameter is a **Value** object containing two parameters:

* *data*: the numerical value of that neural network weight or bias.

* *grad*: the gradient that has been computed by backward propagation.

Note that printing a **DenseNN** object allows you to get a nice representation of it. It has to be read from left to right and it shows the structure of every layer of the neural network. For example:

    [Layer of [ReLUNeuron(3), ReLUNeuron(3)], Layer of [TanhNeuron(2)]]
mean that the first layer is made of:

    2 neuron with ReLU activation, each one with 3 input neurons

and the second layer is made of:

    1 neuron with tanh activations and it takes 2 neurons (previous layer dimension) as inputs

## Training and Predictions

After initializing the neural network, essentially you have to use only 2 other functions: ```nn.Train()``` and ```nn.Predict()```:

* *nn.Train()* takes as inputs:
  * *inputs*: a list containing inputs value to feed in the ```nn``` neural network in order to train it. Note that inputs must be a 2-dimensional array (i.e. ```[[1, 2, 3, ...], [4, 5, 6, ...], [7, 8, 9, ...], ...]```).

  * *predictions*: a list containing predictions to feed in the ```nn``` neural network in order to train it. Note that inputs must be a 1-dimensional array (i.e. ```[1, 2, 3, ...]```).

  * *loss*: a string containing the name of the loss function that has to be used in order to train the neural network. Currently only 1-output neural network are supported and so available loss functions are only: ```MeanSquaredError``` and ```BinaryCrossEntropy```. By default ```loss``` is set to ```''```. Since no loss function is applied by default, make sure to set properly this parameter.

  * *epochs_number*: an integer that specifies the number of training steps. By default is set to ```20``` (an arbitrarily chosen number) so make sure to set it properly.
  
  * *batch_size*: an integer that specifies the batch size for gradient descent (hence how many examples the training function has to take before updating the weights and biases). For example ```batch_size=1``` corresponds to stochastic gradient descent, while ```batch_size=len(inputs)``` corresponds to batch gradient descent (every number between ```1``` and ```len(inputs)``` corresponds to mini batch gradient descent).
  
  * *learning_rate*: a float containing the learning rate for the weights update (```param = param - learning_rate * gradient```). By default is set to ```0.03``` (an arbitrarily chosen number) so make sure to set it properly.

  * *learning_rate_decay*: a float containing the decay rate of learning rate decay (```learning_rate = learning_rate/(1 + learning_rate_decay * epoch)```). By default is set to 0, so no learning rate decay is done without setting this parameter. Note that high ```learning_rate_decay``` makes learning rate decays faster.

* *nn.Train()* gives as output: a **DenseNN** object

* *nn.Predict()* takes as inputs:

  * *input*: a list containing one training set sample that will be fed in the neural network. Note that ```input``` length must match the size of ```nn``` zero-th layer.
* *nn.Predict()* gives as output the final node (a **Value** object) of the ```nn``` neural network containing the prediction value (```nn.Predict(input).data```). A nice graphical representation can also be shown by using the ```nn.graph()``` method of Value objects. For example:  ```nn.Predict(input).graph()```.
