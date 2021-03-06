import numpy as np

class nnet:
    # Initialization function. Assigns random weights for the given network and stores them as list of matrices
    def __init__(self, size):
        self.size=size
        self.num_layers=len(size)
	self.nodes=[np.zeros([l+1, 1]) for l in size[:-1]]
	self.nodes.append(np.zeros([size[-1], 1]))
        self.weights=[np.random.randn(l1+1, l2+1) for (l1, l2) in zip(size[1:-1], size[:-2])]
	self.weights.append(np.random.randn(size[-1], size[-2]+1))


    # function that returns some info about the network. Should include weights in to print
    def __repr__(self):
        return "This is a dnn type with %s layers and %s as unit dimensions" % (str(self.num_layers), str(self.size))
    
    

    def SGD(self, train, test, n_mini_batch, epochs, learn_rate):
	train_samples=len(train)
	print train_samples
	shuff_ind=np.random.shuffle(range(train_samples))
	for epoch in range(epochs):
	    print "Running epoch number %s" % (str(epoch))
	    for k in range(train_samples/n_mini_batch):
	        mini_batch=train[(k-1)*n_mini_batch:k*n_mini_batch]
		self.BackProp(mini_batch, learn_rate)
	    print "Accuracy for Epoch %s is %s" % (str(epoch), str(self.test(test)))
    

    # feed forward function that will be called in Stochastic Gradient Descent process 
    def FeedForward(self, data):
        if np.shape(self.weights[0])[1]!=len(data)+1:
            print "Please check your data dimensions, expected %s but got %s" % (str(np.shape(self.weights[0])[1]), str(len(data)+1))
            raise ValueError('Are you screwing with your data?')
        self.nodes[0][:-1]=np.reshape(data[:], (784, 1))
        for conn in range(len(self.weights)):
            self.nodes[conn+1]=self.sigmoid(np.dot(self.weights[conn], self.nodes[conn]))
    
    # Backpropogation step for a mini batch
    def BackProp(self, mini_batch, learn_rate):
        #first we will find error in outputs
	n=self.num_layers
	del_w=[np.zeros(w_mat.shape) for w_mat in self.weights]
	for case in range(len(mini_batch)):
	    self.FeedForward(mini_batch[case][0])
	    del_this_layer=self.MSE(self.nodes[-1], mini_batch[case][1])*self.nodes[-1]*(1-self.nodes[-1])
	    for l in range(1, n):
		del_w[n-l-1]+=np.dot(del_this_layer, np.transpose(self.nodes[n-l-1]))
	    	del_this_layer=np.dot(np.transpose(self.weights[n-l-1]), del_this_layer)*self.nodes[n-l-1]*(1-self.nodes[n-l-1])
	for l in range(n-1):
	    self.weights[l]-=learn_rate*del_w[l]    	
    

    def test(self, test):
	samples=len(test)
	correct=0
	for case in range(samples):
	    self.FeedForward(test[case][0])
	    correct+=int(np.argmax(self.nodes[-1])==np.argmax(test[case][1]))
        return correct
    # Cost function derivatives		 
    def MSE(self, output, target):
        return output-target
    #Sigmoid function
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
