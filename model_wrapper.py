# Pytorch
import copy
from torch.autograd import grad


class ModelWrapper():
    """ Simple model wrapper to hold pytorch models plus set up the needed
    hooks to access the activations and grads.
    """

    def __init__(self, model=None, bottlenecks={}):
        """ Initialize wrapper with model and set up the hooks to the bottlenecks.

        Args:
            model (nn.Module): Model to test
            bottlenecks (dict): Dictionary attaching names to the layers to
                hook into. Expects, at least, an input, logit and prediction.
        """
        self.ends = None
        self.y_input = None
        self.loss = None
        self.bottlenecks_gradients = None
        self.bottlenecks_tensors = {}
        self.model = copy.deepcopy(model)

        def save_activation(name):
            """ Creates hooks to the activations
            Args:
                name (string): Name of the layer to hook into
            """
            def hook(mod, inp, out):
                """ Saves the activation hook to dictionary
                """
                self.bottlenecks_tensors[name] = out
            return hook

        for name, mod in self.model._modules.items():
            if name in bottlenecks.keys():
                mod.register_forward_hook(save_activation(bottlenecks[name]))
                    
    def _make_gradient_tensors(self, y, bottleneck_name):
        """
        Makes gradient tensor for logit y w.r.t. layer with activations
        
        Args:
            y (int): Index of logit (class)
            bottleneck_name (string): Name of layer activations

        Returns:
            (torch.tensor): Gradients of logit w.r.t. to activations

        """
        acts = self.bottlenecks_tensors[bottleneck_name]
        return grad(self.ends[:, y], acts)

    def eval(self):
        """ Sets wrapped model to eval mode as is done in pytorch.
        """
        self.model.eval()

    def train(self):
        """ Sets wrapped model to train mode as is done in pytorch.
        """        
        self.model.train()

    def __call__(self, x):
        """ Calls prediction on wrapped model pytorch.
        """
        self.ends = self.model(x)
        return self.ends
            
    def get_gradient(self, y, bottleneck_name):
        """ Returns the gradient at a given bottle_neck.

        Args:
            y: Index of the logit layer (class)
            bottleneck_name: Name of the bottleneck to get gradients w.r.t.

        Returns:
            (torch.tensor): Tensor containing the gradients at layer.
        """
        self.y_input = y
        return self._make_gradient_tensors(y, bottleneck_name)
