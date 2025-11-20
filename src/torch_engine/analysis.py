import torch
import numpy as np

def get_model_params(model):
    """
    Extracts weights and biases from the ExplainableReLUNet as numpy arrays.
    """
    params = {}
    # Layer 1 (Hidden)
    params['w_in'] = model.hidden.weight.detach().cpu().numpy()  # (Hidden, 1)
    params['b_in'] = model.hidden.bias.detach().cpu().numpy()    # (Hidden,)
    
    # Layer 2 (Output)
    params['w_out'] = model.output.weight.detach().cpu().numpy() # (1, Hidden)
    params['b_out'] = model.output.bias.detach().cpu().item()    # Scalar
    
    return params

def compute_kinks(model):
    """
    Calculates the 'Kink Points' (x-coordinates) where each ReLU neuron turns on.
    
    ReLU(w*x + b) turns on when w*x + b = 0
    => x_kink = -b / w
    
    Returns:
        kinks (np.array): x-coordinates of kinks, sorted.
        indices (np.array): Original neuron indices corresponding to sorted kinks.
    """
    params = get_model_params(model)
    w_in = params['w_in'].flatten()
    b_in = params['b_in'].flatten()
    
    # Avoid division by zero
    epsilon = 1e-7
    
    # Calculate kinks
    kinks = -b_in / (w_in + epsilon)
    
    # Sort them for analysis
    sorted_indices = np.argsort(kinks)
    sorted_kinks = kinks[sorted_indices]
    
    return sorted_kinks, sorted_indices

def extract_basis_functions(model, x_range=(-1, 7), resolution=1000):
    """
    Analytically reconstructs the basis functions without running a forward pass
    on data, useful for theoretical plotting.
    
    Returns:
        x (np.array): Domain.
        bases (np.array): Shape (N, Hidden).
    """
    params = get_model_params(model)
    w_in = params['w_in'].flatten() # (H,)
    b_in = params['b_in'].flatten() # (H,)
    w_out = params['w_out'].flatten() # (H,)
    
    x = np.linspace(x_range[0], x_range[1], resolution)
    
    # Compute w_out * ReLU(w_in * x + b_in)
    # Using broadcasting: x is (N, 1), w_in is (1, H)
    z = x[:, None] * w_in[None, :] + b_in[None, :] # (N, H)
    activations = np.maximum(0, z)
    weighted_bases = activations * w_out[None, :] # (N, H)
    
    return x, weighted_bases
