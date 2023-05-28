import numpy as np
from .module import Module

class Batchnorm(Module):
    """
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    """
    
    def __init__(self, inFeatures, learningRate, eps = 1e-5, momentum = 0.9):
        super(Batchnorm, self).__init__()
        self.gamma = dict()
        self.beta = dict()
        self.mode = 'train'
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(inFeatures)
        self.running_var = np.zeros(inFeatures)
        self.cache = dict()
        self.lr = learningRate

        self.gamma["grad"] = None
        self.beta["grad"] = None
        self.gamma["val"] = np.ones(inFeatures)
        self.beta["val"] = np.zeros(inFeatures)
        self.cache['input'] = None

    def forward(self, x):
        self.cache['input'] = x
        gamma = self.gamma['val']
        beta = self.beta['val']

        if self.mode == 'train':
            sample_mean = np.mean(x,0)
            sample_var = np.var(x,0)
            norm_x = (x - sample_mean) / np.sqrt(sample_var + self.eps)
            out = gamma * norm_x + beta

            self.running_mean = self.momentum * self.running_mean + \
                (1 - self.momentum) * sample_mean
            self.running_var = self.momentum * self.running_var + \
                (1 - self.momentum) * sample_var

            self.cache['sample_mean'] = sample_mean
            self.cache['sample_var'] = sample_var
            self.cache['norm_x'] = norm_x

        elif self.mode == 'test':
            norm_x = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = gamma * norm_x + beta

        return out
    

    def backward(self, dout):
        x = self.cache['input']
        sample_mean = self.cache['sample_mean']
        sample_var = self.cache['sample_var']
        norm_x = self.cache['norm_x']
        gamma = self.gamma['val']

        N = x.shape[0]
        self.beta["grad"] = np.sum(dout, 0)
        self.gamma["grad"] = np.sum(dout * norm_x, 0)
        A = dout * gamma
        Z = x - sample_mean
        B = -Z / (sample_var + self.eps)
        C = 1 / np.sqrt(sample_var + self.eps)
        D = ((sample_var + self.eps) ** (-1/2))/2
        BA = np.sum(B * A, 0)

        dx1 = 2 / N * Z * D * BA
        dx2 = 1 / N * (np.sum(-2 / N * Z * D * BA, 0) - np.sum(C * A, 0))
        dx3 = C * A
        dx = dx1 + dx2 + dx3
        return dx
    
    def step(self):
        #weight and bias values update
        self.gamma["val"] = self.gamma["val"] - self.lr*self.gamma["grad"]
        self.beta["val"] = self.beta["val"] - self.lr*self.beta["grad"]
        return
    
    
    def num_params(self):
        #total number of trainable parameters in the layer
        numParams = self.gamma["val"].shape[0]+ self.beta["val"].shape[0]
        return numParams