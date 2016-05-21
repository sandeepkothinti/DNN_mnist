import cPickle, gzip
import numpy as np
def load():
  f=gzip.open('mnist.pkl.gz')
  train, valid, test=cPickle.load(f)
  train_lab=[vectorized(i) for i in train[1]]
  valid_lab=[vectorized(i) for i in valid[1]]
  test_lab=[vectorized(i) for i in test[1]]
  train=zip(train[0], train_lab)
  valid=zip(valid[0], valid_lab)
  test=zip(test[0], test_lab)
  f.close()
  return (train, valid, test)
def vectorized(inp):
    temp=np.zeros((10,1))
    temp[inp]=1
    return temp
 
