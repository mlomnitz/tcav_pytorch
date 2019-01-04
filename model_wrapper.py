# Python
import numpy as np
# Pytorch
import torch


class ModelWrapper():
    """ Simple model wrapper to hold pytorch models plus set up the needed
    hooks to access the activations and grads
    """

    def __init__(self, model=None, bottlenecks={}):
        """ Initialize wrapper with model and set up the hooks to the bottlenecks

        Args:
            model (nn.Module): Model to test
            bottlenecks (dict): Dictionary attaching names to the layers to
                hook into. Expects, at least, an input, logit and prediction.
        """
        self.model = model
        self.grads = {}
        if 'logits' or 'input' or 'prediction' not in bottlenecks.keys():
            raise 'Wrapper expects at least logits, input and predictions'

        self.bottleneck = bottlenecks
        
        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad
            return hook
        
        for key in self.bottleneck.keys():
            self.model.register_hook(save_grad(self.bottleneck[key]))
        
    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()
        
