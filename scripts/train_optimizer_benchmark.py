import sys
import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import setup_logger, set_seed, get_base_args
from src.data_loader import load_mnist, generate_spiral_data
from src.visualization import plot_training_curves
from src import numpy_engine as nn

def build_model(input_dim, output_dim, hidden_dims=[128, 64]):
    """Constructs a NumpyMLP with specified architecture."""
    model = nn.NumpyMLP()
    
    prev_dim = input_dim
    for h_dim in hidden_dims:
        model.add(nn.Linear(prev_dim, h_dim))
        model.add(nn.ReLU())
        prev_dim = h_dim
        
    model.add(nn.Linear(prev_dim, output_dim))
    # Softmax is handled implicitly by CrossEntropyLoss usually, 
    # or explicitly if the loss expects probabilities. 
    # Our CrossEntropy expects raw logits (Z).
    return model

def train_one_epoch(model, optimizer, loss_fn, x_train, y_train, batch_size):
    num_samples = x_train.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    total_loss = 0
    
    # Shuffle
    indices = np.random.permutation(num_samples)
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]
    
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_samples)
        
        batch_x = x_shuffled[start:end]
        batch_y = y_shuffled[start:end]
        
        # 1. Forward
        preds = model.forward(batch_x)
        
        # 2. Loss
        loss = loss_fn.forward(preds, batch_y)
        total_loss += loss
        
        # 3. Backward
        grad = loss_fn.backward()
        model.backward(grad)
        
        # 4. Step
        optimizer.step()
        optimizer.zero_grad()
        
    return total_loss / num_batches

def run_benchmark(args, config):
    logger = setup_logger("OptimizerBenchmark", f"{config['experiment']['output_dir']}/benchmark.log")
    set_seed(args.seed)
    
    # 1. Load Data
    if args.dataset == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist(
            train_size=config['data']['mnist']['train_size'],
            test_size=config['data']['mnist']['test_size']
        )
        input_dim = 784
        output_dim = 10
    elif args.dataset == 'spiral':
        # For spiral, we need to one-hot encode manually for the Numpy Engine's CrossEntropy
        x_raw, y_raw = generate_spiral_data(
            n_points=config['data']['spiral']['n_points'],
            K=config['data']['spiral']['K'],
            sigma=config['data']['spiral']['sigma']
        )
        # One-hot encode
        from tensorflow.keras.utils import to_categorical
        y_encoded = to_categorical(y_raw, num_classes=config['data']['spiral']['K'])
        
        # Simple split
        split = int(0.8 * len(x_raw))
        x_train, x_test = x_raw[:split], x_raw[split:]
        y_train, y_test = y_encoded[:split], y_encoded[split:]
        
        input_dim = 2
        output_dim = config['data']['spiral']['K']
    else:
        raise ValueError("Dataset must be 'mnist' or 'spiral'")

    # 2. Determine Optimizers to Run
    optimizers_to_run = ['sgd', 'momentum', 'adam'] if args.optimizer == 'all' else [args.optimizer]
    
    history = {}
    
    for opt_name in optimizers_to_run:
        logger.info(f"Starting training with {opt_name.upper()}...")
        
        # Re-init model for fairness
        model = build_model(input_dim, output_dim)
        loss_fn = nn.CrossEntropyLoss()
        
        # Setup Optimizer
        params = model.parameters()
        opt_config = config['optimizer_params'].get(opt_name, {})
        
        if opt_name == 'sgd':
            optimizer = nn.SGD(params, lr=opt_config.get('lr', 0.01))
        elif opt_name == 'momentum':
            optimizer = nn.Momentum(params, lr=opt_config.get('lr', 0.01), momentum=opt_config.get('momentum', 0.9))
        elif opt_name == 'adam':
            optimizer = nn.Adam(
                params, 
                lr=opt_config.get('lr', 0.001),
                beta1=opt_config.get('beta1', 0.9),
                beta2=opt_config.get('beta2', 0.999),
                epsilon=float(opt_config.get('epsilon', 1e-8))
            )
            
        # Training Loop
        losses = []
        epochs = config['training']['epochs']
        batch_size = config['training']['batch_size']
        
        for epoch in range(epochs):
            avg_loss = train_one_epoch(model, optimizer, loss_fn, x_train, y_train, batch_size)
            losses.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                logger.info(f"[{opt_name.upper()}] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        history[opt_name] = losses
        
        # Evaluation
        test_preds = model.forward(x_test)
        acc = np.mean(np.argmax(test_preds, axis=1) == np.argmax(y_test, axis=1))
        logger.info(f"[{opt_name.upper()}] Final Test Accuracy: {acc*100:.2f}%")

    # 3. Plot Results
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)
    plot_path = f"{config['experiment']['output_dir']}/optimizer_comparison_{args.dataset}.png"
    
    plot_training_curves(history, title=f"Optimizer Convergence ({args.dataset.upper()})")
    plt.savefig(plot_path)
    logger.info(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    parser = get_base_args()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'spiral'])
    parser.add_argument('--optimizer', type=str, default='all', choices=['sgd', 'momentum', 'adam', 'all'])
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    run_benchmark(args, config)
