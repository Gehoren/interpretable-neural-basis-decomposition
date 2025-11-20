from .utils import setup_logger, set_seed
from .data_loader import generate_sine_wave, generate_spiral_data, PolynomialFeatureExpander, load_mnist
from .visualization import plot_basis_mechanisms, plot_decision_boundary, plot_training_curves

# Expose engines
from . import numpy_engine
from . import torch_engine
