import numpy as np

class Activation:
    """Base class for activation functions."""
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
    def __str__(self):
        return self.__class__.__name__

class ReLU(Activation):
    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        # dL/dx = dL/dy * dy/dx
        # dy/dx = 1 if x > 0 else 0
        return grad_output * (self.cache > 0)

class Sigmoid(Activation):
    def forward(self, x):
        # Numerically stable sigmoid
        self.output = np.where(
            x >= 0, 
            1 / (1 + np.exp(-x)), 
            np.exp(x) / (1 + np.exp(x))
        )
        return self.output
    
    def backward(self, grad_output):
        # dy/dx = y * (1 - y)
        return grad_output * self.output * (1 - self.output)

class Tanh(Activation):
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad_output):
        # dy/dx = 1 - y^2
        return grad_output * (1 - self.output ** 2)

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.cache = x
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, grad_output):
        dx = np.ones_like(self.cache)
        dx[self.cache < 0] = self.alpha
        return grad_output * dx

class SELU(Activation):
    """
    Scaled Exponential Linear Unit. 
    Self-normalizing activation function (Klambauer et al., 2017).
    """
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def forward(self, x):
        self.cache = x
        return self.scale * np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def backward(self, grad_output):
        x = self.cache
        gradient = np.where(x > 0, self.scale, self.scale * self.alpha * np.exp(x))
        return grad_output * gradient

class Softmax(Activation):
    """
    Softmax activation. 
    Note: Usually combined with CrossEntropy for numerical stability in gradients,
    but implemented here standalone for modularity.
    """
    def forward(self, x):
        # Shift x for numerical stability (subtract max)
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shift_x)
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output
    
    def backward(self, grad_output):
        # Note: This is a simplified element-wise backward pass approximation 
        # often used when Softmax is not directly coupled with CrossEntropy in the engine.
        # For true Jacobian rigor, tensor dimensions increase. 
        # Here we assume standard use case where grad_output comes from a loss 
        # that handles the heavy lifting (like CrossEntropy) or simplified propagation.
        
        # If strictly computing dSoftmax/dx * grad:
        # (S_i * (delta_ij - S_j)) 
        # We implement the vector version for batch efficiency:
        # grad_input = output * (grad_output - sum(grad_output * output))
        
        s = self.output
        return s * (grad_output - np.sum(grad_output * s, axis=1, keepdims=True))
