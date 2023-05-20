import numpy as np
from .module import Module  

#definition for the Cross Entropy Loss layer class
class CrossEntropyLoss(Module):
    
    """
    Computes the cross entropy loss using the output and the required target class.
    loss = -log(yHat[class])
    
    Shapes:
    Outputs - (N,C)
    Classes - (N)
    Out Gradients - (N,C)
    """
    
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        #declaring and initializing the cache dictionary
        self.cache = dict()
        self.cache["scores"] = None
        self.cache["classes"] = None
        self.cache["numClasses"] = None
        return
    
    
    def forward(self, outputs, classes):
        #computing the loss for each sample in the batch and averaging the loss
        scores = outputs[np.arange(outputs.shape[0]), classes]
        loss = -np.mean(np.log(scores))
        
        #storing the scores and classes in cache
        self.cache["scores"] = scores
        self.cache["classes"] = classes
        self.cache["numClasses"] = outputs.shape[1]
        return loss
    
    
    def backward(self):
        #computing the loss gradients
        N = len(self.cache["classes"])
        gradients = np.zeros((N, self.cache["numClasses"]))
        gradients[np.arange(N), self.cache["classes"]] = -1./self.cache["scores"]
        gradients /= N
        return gradients