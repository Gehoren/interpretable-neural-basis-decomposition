import numpy as np

class Optimizer:
    """Base class for optimizers."""
    def __init__(self, params, lr=0.01):
        self.params = params  # List of dicts: [{'data': W, 'grad': dW, ...}]
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        """Sets gradients to zero."""
        for p in self.params:
            p['grad'].fill(0)

class SGD(Optimizer):
    """Stochastic Gradient Descent."""
    def step(self):
        for p in self.params:
            # W = W - lr * dW
            p['data'] -= self.lr * p['grad']

class Momentum(Optimizer):
    """SGD with Momentum."""
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params, lr)
        self.momentum = momentum
        # Initialize velocity for each parameter
        for p in self.params:
            p['velocity'] = np.zeros_like(p['data'])

    def step(self):
        for p in self.params:
            # v = momentum * v - lr * grad (PyTorch style usually: v = m*v + g; p = p - lr*v)
            # Standard Momentum formula: v = beta * v + (1 - beta) * grad  <-- Provided in prompt specs
            # The prompt specified: v = beta * v + (1-beta) * grad
            #                       w = w - eta * v
            
            p['velocity'] = self.momentum * p['velocity'] + (1 - self.momentum) * p['grad']
            p['data'] -= self.lr * p['velocity']

class Adam(Optimizer):
    """Adaptive Moment Estimation."""
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step
        
        # Initialize moments
        for p in self.params:
            p['m'] = np.zeros_like(p['data']) # First moment
            p['v'] = np.zeros_like(p['data']) # Second moment

    def step(self):
        self.t += 1
        for p in self.params:
            grad = p['grad']
            
            # Update biased first moment estimate
            p['m'] = self.beta1 * p['m'] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            p['v'] = self.beta2 * p['v'] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = p['m'] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = p['v'] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p['data'] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
