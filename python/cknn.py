"""
Contains a class for UKF-training a feedforward neural-network.
This is primarily to demonstrate the advantages of UKF-training.
See the class docstrings for more details.
This module also includes a function for loading stored KNN objects.
"""
from __future__ import division
import numpy as np; npl = np.linalg
from scipy.linalg import block_diag
from time import time
import pickle

##########

def load_knn(filename):
    """
    Loads a stored KNN object saved with the string filename.
    Returns the loaded object.
    """
    if not isinstance(filename, str):
        raise ValueError("The filename must be a string.")
    if filename[-4:] != '.knn':
        filename = filename + '.knn'
    with open(filename, 'rb') as input:
        W, neuron, P = pickle.load(input)
    obj = CKNN(W[0].shape[1]-1, W[1].shape[0], W[0].shape[0], neuron)
    obj.W, obj.P = W, P
    return obj

##########

class CKNN:
    """
    Class for a feedforward neural network (NN). Currently 
    is always fully-connected, and uses the same activation function type for every neuron.
    The NN can be trained by cubature kalman filter (CKF) or stochastic gradient descent (SGD).
    Use the train function to train the NN, the feedforward function to compute the NN output,
    and the classify function to round a feedforward to the nearest class values. A save function
    is also provided to store a KNN object in the working directory.
    """
    def __init__(self, neurons, neuron_type, sprW=5):
        """
            neurons: vector of nodes starting from the input layer to the output layer 
            neuron_type: activation function type; 'logistic', 'tanh', or 'relu'
            sprW: spread of initial randomly sampled synapse weights; float scalar
        """
        # Function dimensionalities
        self.nu = int(neurons[0])           #dimensionality of input  
        self.ny = int(neurons[-1])          #dimensionality of output
        self.neurons = map(int, neurons)    #store the integer of the nodes of NN structure
        self.num_layers = len(neurons)-1
        
        
        # Neuron type
        if neuron_type == 'logistic':
            self.sig = lambda V: (1 + np.exp(-V))**-1
        elif neuron_type == 'tanh':
            self.sig = lambda V: np.tanh(V)
        elif neuron_type == 'relu':
            self.sig = lambda V: np.clip(V, 0, np.inf)
        else:
            raise ValueError("The neuron argument must be 'logistic', 'tanh', or 'relu'.")
        self.neuron_type = neuron_type

        # Initial synapse weight matrices
        sprW = np.float64(sprW)
        self.W  = []    #the weights
        [self.W.append(sprW*(2*np.random.sample((self.neurons[i+1], self.neurons[i]))-1)) for i in range(self.num_layers)]
        self.b = []     #the bias
        [self.b.append(sprW*(2*np.random.sample((self.neurons[i+1], 1))-1)) for i in range(self.num_layers)]
        self.nparams = np.dot(self.neurons[:-1],self.neurons[1:]) + sum(self.neurons[1:])
        self.P = None

        # Function for pushing signals through a synapse with bias
        self._affine_dot = lambda W, b, V: np.dot(np.atleast_1d(V), W.T) + b

        # Function for computing the RMS error of the current fit to some data set
        self.compute_rms = lambda U, Y: np.sqrt(np.mean(np.square(Y - self.feedforward(U))))

####

    def save(self, filename):
        """
        Saves the current NN to a file with the given string filename.
        """
        if not isinstance(filename, str):
            raise ValueError("The filename must be a string.")
        if filename[-4:] != '.knn':
            filename = filename + '.knn'
        with open(filename, 'wb') as output:
            pickle.dump((self.W, self.neuron, self.P), output, pickle.HIGHEST_PROTOCOL)

####

    def feedforward(self, U, get_l=False):
        """
        Feeds forward an (m by nu) array of inputs U through the NN.
        Returns the associated (m by ny) output matrix, and optionally
        the intermediate activations l.
        """
        U = np.float64(U)
        if U.ndim == 1 and len(U) > self.nu: U = U[:, np.newaxis]
        l = []
        temp = U
        for i in range(self.num_layers-1):
            temp = self.sig(self._affine_dot(self.W[i], self.b[i], temp))
            l.append(temp)        
        h = self._affine_dot(self.W[-1], self.b[-1], temp)
        
        if get_l: return h, l
        return h

####