import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mnist_load
import network
#train, valid, test=mnist_load.load()

#train_feat=np.array(train[0])
#train_label=np.array(train[1])
#valid_feat=np.array(valid[0])
#valid_label=np.array(valid[1])
#test_feat=np.array(test[0])
#test_label=np.array(test[1])
net_dnn=network.nnet([2, 2, 3])
print net_dnn
out=net_dnn.feedforward([[1], [2]])
print out
def plot_mnist_digit(image):
    """ Plot a single MNIST image."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()


#plot_mnist_digit(np.reshape(train_feat[0], (28, 28)))

