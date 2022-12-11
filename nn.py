from engine import Value
import random

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin):
        self.w = [Value(random.gauss(0, 1/nin), label = 'weight') for _ in range(nin)] #Xavier initialization
        self.b = Value(0, label = 'bias')
        self.nonlin = nonlin

    def __call__(self, x):
        act=()
        for wi, xi in zip(self.w, x):
            act += (wi*xi, )
        act += (self.b, )
        act = act[0].multisum(act[1:])
        
        self.nonlin = self.nonlin.lower()
        
        assert self.nonlin in ('', 'linear', 'sigmoid', 'relu', 'tanh'), f"{self.nonlin} is not a valid activation function. Valid activation functions are: linear, relu, sigmoid, tanh"
        
        # act = act if self.nonlin == ('linear' or '') else act
        act = act.relu() if self.nonlin == 'relu' else act
        act = act.sigmoid() if self.nonlin == 'sigmoid' else act
        act = act.tanh() if self.nonlin == 'tanh' else act

        return act
        
    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        nonlin = self.nonlin.capitalize() #just for nicer visualization
        nonlin = 'ReLU' if self.nonlin.lower() == 'relu' else nonlin #just for nicer visualization
            
        return f"{nonlin}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, activation):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class DenseNN(Module):

    def __init__(self, nin, nouts, activations):
        
        assert isinstance(nin, int), "nin must be int"
        assert isinstance(nouts, (tuple, list)), "nouts must be tuple or list"
        assert isinstance(activations, (tuple, list)), "activations must be tuple or list"
        assert len(nouts) == len(activations), f"layers and activations must have the same size. Size of layers is {len(nouts)} whilst size of activations is {len(activations)}"
        assert nouts[len(nouts)-1] == 1, "multi-output NN is not supported: last layer must consist of just 1 neuron"
        
        sz = (nin, ) + nouts if isinstance(nouts, tuple) else sz
        sz = [nin] + nouts if isinstance(nouts, list) else sz
                
        self.layers = [Layer(sz[i], sz[i+1], activations[i]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def Predict(self, input):
        assert isinstance(input, list) and isinstance(input[0], (int, float)), "inputs must be a 1-dimensional list of ints or floats"
        assert len(self.layers[0].neurons[0].w) == len(input), f"inputs lenght must match first 0-th layer size. 0-th layer has dimension {len(self.layers[0].neurons[0].w)} whilst inputs has size {len(input)}"
        
        return self(input)
    
    def Train(self, inputs, predictions, loss, epochs_number, batch_size,
          learning_rate, learning_rate_decay = 0):
        
        def update_parameters(self, learning_rate, batch_size, epoch, current_example, learning_rate_decay, example_set_size):
            learning_rate=(1/(1+learning_rate_decay*epoch))*learning_rate
            current_example += 1
            if (current_example % batch_size == 0) or (current_example==example_set_size):
                if (current_example==example_set_size) and (current_example % batch_size != 0):
                    batch_size = example_set_size % batch_size
                for parameter in self.parameters():
                    parameter.data -= (learning_rate/batch_size) * parameter.grad
        
        assert isinstance(inputs, list) and isinstance(inputs[0], list) and isinstance(inputs[0][0], (int, float)), "inputs must be 2-dimensional list of ints or floats"
        assert isinstance(predictions, list) and isinstance(predictions[0], (int, float)), "predictions must be 1-dimensional list of ints or floats"

        loss = loss.upper()
        
        assert loss == 'MSE' or loss == 'BCE' or loss == '', "loss function is not valid"
        
        if loss == 'MSE':
            loss_func = lambda n, x : n.MeanSquaredError(x)
        elif loss == 'BCE':
            loss_func = lambda n, x : n.BinaryCrossEntropy(x)
        else:
            loss_func = lambda x, y : x
        
        print("Training neural network...")
        print("Epoch", end=' ')
        for epoch in range(epochs_number):
            print(f"#{epoch+1}, ", end='')
            for example in range(len(inputs)):
                out = self(inputs[example])
                out = loss_func(out, predictions[example])
                out.backward()
                update_parameters(self, learning_rate, batch_size, epoch, example, learning_rate_decay, len(inputs))
                self.zero_grad()
                
        print("\nNeural network successfully trained.")
        return self

    def __repr__(self):
        return f"NeuralNetwork of [{', '.join(str(layer) for layer in self.layers)}]"
    
