import numpy as np
from .module import Module

class Dropout(Module):
    """ 
    - x: Input data, of any shape
    - p: Dropout parameter. We keep each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not
      in real networks.
    """    
    
    def __init__(self, p, seed = None):
        super(Dropout, self).__init__()
        self.mode = 'train'
        self.p = p
        if seed is not None:
            np.random.seed(seed)
        self.cache = dict()


    def forward(self, x):
        mask = None
        if self.mode == 'train':
            mask = (np.random.rand(*x.shape) < self.p) / self.p
            out = x * mask

        elif self.mode == 'test':    
            out = x

        self.cache['mask'] = mask
        out = out.astype(x.dtype, copy=False)

        return out
    
    def backward(self, dout):
        if self.mode == 'train':
            dx = self.cache['mask'] * dout
        elif self.mode == 'test':
            dx = dout
        return dx