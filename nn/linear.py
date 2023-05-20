import numpy as np
from .module import Module  

class Linear(Module):
    
    """
    Fully Connected layer that applies a linear transformation to the input using weights and biases.
    y = w.x + b
    
    Shapes:
    Input - (N, Fin)
    Output - (N, Fout)
    Out Gradients - (N, Fout)
    In Gradients - (N, Fin)
    """
    
    def __init__(self, inFeatures, outFeatures, learningRate):
        super(Linear, self).__init__()
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.lr = learningRate
        
        #declaring the weight and bias dictionaries for storing their values and gradients
        self.w = dict()
        self.b = dict()
        
        #initializing the weight and bias values and gradients
        self.w["grad"] = None
        self.b["grad"] = None
        self.w["val"] = np.random.uniform(-np.sqrt(1/inFeatures), np.sqrt(1/inFeatures), size=(outFeatures, inFeatures))
        self.b["val"] = np.random.uniform(-np.sqrt(1/inFeatures), np.sqrt(1/inFeatures), size=(outFeatures))
        
        #declaring and initializing the cache dictionary
        self.cache = dict()
        self.cache["input"] = None
        return
    
    
    def forward(self, inputBatch):
        #computing the linear transformation and storing the inputs in the cache
        outputBatch = np.dot(inputBatch, self.w["val"].T) + self.b["val"]
        self.cache["input"] = inputBatch
        return outputBatch
    
    
    def backward(self, gradients):
        #computing the gradients wrt the weight
        [N, Fin] = self.cache["input"].shape
        # wGrad = np.einsum('no,ni->noi', gradients, self.cache["input"])
        # self.weight["grad"] = np.mean(wGrad, axis=0)
        self.w["grad"] = np.dot(gradients.T, self.cache["input"])
        
        #computing the gradients wrt the bias
        # bGrad = np.dot(gradients, np.eye(gradients.shape[1]))
        # self.bias["grad"] = np.mean(bGrad, axis=0)
        self.b["grad"] = gradients.sum(axis= 0)
        
        #computing the gradients wrt the input
        # inGrad = self.weight["val"]
        # return np.dot(gradients, inGrad)
        dzdx = np.dot(gradients, self.w["val"])
        return dzdx
    
    
    def step(self):
        #weight and bias values update
        self.w["val"] = self.w["val"] - self.lr*self.w["grad"]
        self.b["val"] = self.b["val"] - self.lr*self.b["grad"]
        return
    
    
    def num_params(self):
        #total number of trainable parameters in the layer
        numParams = (self.w["val"].shape[0]*self.w["val"].shape[1]) + self.b["val"].shape[0]
        return numParams