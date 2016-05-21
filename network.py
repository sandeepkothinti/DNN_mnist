import numpy as np

class nnet:
    # Initialization function. Assigns random weights for the given network and stores them as list of matrices
    def __init__(self, size):
        self.size=size
        self.num_layers=len(size)
        self.weights=[]
        self.biases=[]
        for (l1, l2) in zip(size[1:], size[:-1]):
            self.weights.append(np.random.randn(l1, l2))
            self.biases.append(np.random.randn(l1, 1))
    # function that returns some info about the network. Should include weights in to print
    def __repr__(self):
        return "This is a dnn type with %s layers and %s as unit dimensions" % (str(self.num_layers), str(self.size))


    # feed forward function that will be called in Stochastic Gradient Descent process 
    def feedforward(self, data):
        if np.shape(self.weights[0])[1]!=len(data):
            print "Please check your data dimensions, expected %s but got %s" % (str(np.shape(self.weights[0])[1]), str(len(data)))
            raise ValueError('Are you screwing with your data?')
        this_layer=data
        for conn in range(len(self.weights)):
            this_layer=self.sigmoid(np.add(np.dot(self.weights[conn],this_layer), self.biases[conn]))
        return this_layer
    def MSE(self, output, target):
        return output-target
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
