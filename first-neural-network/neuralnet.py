import numpy as np


class ReLu:
    def __init__(self, hidden_units = 10, input_size = 10, rand_seed=123, layer_name=''):
        np.random.seed(rand_seed)
        self.weights = np.random.uniform(low=-1, high=1, size = (input_size, hidden_units))
        self.a = np.zeros(shape=(hidden_units, 1)) # column vector; will need to reuse
        self.layer_name = 'relu_layer' if len(layer_name) < 1 else layer_name

    def _f(self, a):
        # a should be an array
        zeros = np.zeros_like(a)
        out = np.concatenate([zeros, a]) # should be a nx2 matrix
        return np.max(out, axis=1) # should be a nx1 matrix
    
    def _df(self, a):
        return np.ones_like(a)
    
    def forward(self, xinput):
        '''
        Input is a covariate matrix X that is dimension nxm
        return a matrix nxp where p is the number of hidden units
        '''
        self.a = np.dot(xinput, self.weights) # check if this is a column vector
        out = self._f(self.a)
        return out

    def deltaRule(self, X, y, predicted):
        '''
        error = y^ - y
        '''
        error = predicted - y
        hidden_error = np.dot(self.weights, error)
        delta = hidden_error * self._df(hidden_error)



        