import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import copy
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class PhotonicNeuralNetwork(nn.Module):
    """
    Simulated photonic neural network with realistic constraints:
    - Limited observability of intermediate activations
    - Thermal crosstalk between neighboring neurons
    - Nonlinear laser dynamics and coherence effects
    - Energy tracking for various operations
    """
    def __init__(self, input_size, hidden_sizes, output_size, 
                 noise_level=0.02, 
                 crosstalk_factor=0.05,
                 coherence_factor=0.03):
        super(PhotonicNeuralNetwork, self).__init__()
        
        # Network architecture
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Create multiple hidden layers
        self.hidden_layers = nn.ModuleList()
        
        # Input to first hidden layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Additional hidden layers
        for i in range(len(hidden_sizes)-1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
        # Physical parameters
        self.noise_level = noise_level
        self.crosstalk_factor = crosstalk_factor
        self.coherence_factor = coherence_factor
        
        # Energy tracking
        self.energy_consumed = 0
        self.energy_probe = 0.05  # Energy cost of probing a neuron
        self.energy_compute = 0.01  # Base energy for forward computation
        self.energy_weight_update = 0.02  # Energy for updating weights
        
        # Thermal state (simulates temperature of photonic elements)
        self.thermal_state = torch.zeros(max(hidden_sizes))
        
        # Additional state for tracking
        self.using_rl = False
        self.activation_traces = []
        
    def reset_thermal_state(self):
        """Reset the thermal state of the system"""
        self.thermal_state = torch.zeros_like(self.thermal_state)
    
    def apply_thermal_effects(self, x, layer_idx):
        """Apply thermal crosstalk effects to a layer's output"""
        # Update thermal state based on activation
        layer_size = x.size(-1)
        layer_thermal = torch.sum(x.abs(), dim=0) * self.crosstalk_factor
        
        # Resize if needed
        if layer_size > len(self.thermal_state):
            old_thermal = self.thermal_state
            self.thermal_state = torch.zeros(layer_size)
            self.thermal_state[:len(old_thermal)] = old_thermal
        
        # Update thermal state (with partial cooling)
        self.thermal_state[:layer_size] = 0.7 * self.thermal_state[:layer_size] + 0.3 * layer_thermal
        
        # Apply thermal crosstalk
        thermal_noise = torch.zeros_like(x)
        for i in range(x.size(-1)):
            # Affect neighboring neurons based on distance
            for j in range(x.size(-1)):
                dist = abs(i - j)
                if dist > 0:
                    thermal_effect = self.thermal_state[i] * self.crosstalk_factor / (dist**2)
                    thermal_noise[:, j] += thermal_effect
        
        return x + thermal_noise
        
    def apply_coherence_effects(self, x):
        """Apply optical coherence effects (phase-dependent interference)"""
        # Simulate phase relationships between neurons
        phase = torch.randn_like(x) * self.coherence_factor
        coherence_effect = x * torch.cos(phase)
        return x + (coherence_effect - x) * self.coherence_factor
    
    def forward(self, x, add_noise=True, record_activations=False):
        """
        Forward pass through the photonic neural network
        
        Parameters:
            x: input tensor
            add_noise: whether to add physical noise
            record_activations: whether to record intermediate activations
        """
        batch_size = x.shape[0]
        activations = []
        
        # Track energy for computation
        self.energy_consumed += batch_size * self.energy_compute
        
        # Input normalization (optical scaling)
        x = torch.tanh(x)  # Bound inputs as optical systems have limited dynamic range
        
        if record_activations:
            activations.append(x.detach().clone())
        
        # Process through hidden layers with photonic effects
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = torch.tanh(x)  # Non-linearity (e.g., optical nonlinearity)
            
            # Add photonic-specific effects when training
            if add_noise and self.training:
                # Physical noise (laser intensity fluctuations)
                x = x + torch.randn_like(x) * self.noise_level
                
                # Thermal crosstalk between neighboring photonic neurons
                x = self.apply_thermal_effects(x, i)
                
                # Coherence effects in optical systems
                x = self.apply_coherence_effects(x)
            
            if record_activations:
                activations.append(x.detach().clone())
                # Energy cost for probing/measuring neuron states
                self.energy_consumed += x.numel() * self.energy_probe
        
        # Output layer
        x = self.output_layer(x)
        
        if record_activations:
            activations.append(x.detach().clone())
            self.activation_traces = activations
            
        return x
    
    def reset_energy_counter(self):
        """Reset the energy consumption counter"""
        self.energy_consumed = 0
        
    def get_energy_consumed(self):
        """Get total energy consumed"""
        return self.energy_consumed


class CovarianceMatrixAdaptation:
    """
    Implementation of Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    Adapted specifically for training photonic neural networks with limited observability
    """
    def __init__(self, model, population_size=16, sigma=0.1, learning_rate=0.1, 
                 energy_constraint=1000, adapt_sigma=True):
        self.model = model
        self.model.using_rl = True
        
        # Get parameter dimensions
        self.n_params = sum(p.numel() for p in model.parameters())
        
        # CMA-ES parameters
        self.population_size = population_size
        self.sigma = sigma  # Step size
        self.learning_rate = learning_rate
        self.energy_constraint = energy_constraint
        self.adapt_sigma = adapt_sigma
        
        # Initialize mean as current parameters
        self.mean = self.get_flat_params()
        
        # Initialize covariance matrix as identity
        self.C = torch.eye(self.n_params)
        
        # Path variables for adaptation
        self.p_sigma = torch.zeros(self.n_params)
        self.p_c = torch.zeros(self.n_params)
        
        # Constants for adaptation
        self.c_sigma = 0.1
        self.c_c = 0.1
        self.c_1 = 0.1
        self.c_mu = 0.1
        
        # History for tracking
        self.fitness_history = []
        self.energy_history = []
        self.sigma_history = []
        
    def get_flat_params(self):
        """Get flattened parameters from model"""
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])
        
    def set_flat_params(self, flat_params):
        """Set flattened parameters to model"""
        idx = 0
        for param in self.model.parameters():
            flat_size = param.numel()
            param.data.copy_(flat_params[idx:idx + flat_size].view(param.shape))
            idx += flat_size
    
    def sample_population(self):
        """Sample a population of parameter vectors"""
        # Generate random samples from multivariate normal distribution
        try:
            # Try to compute Cholesky decomposition
            L = torch.linalg.cholesky(self.C)
            z_samples = torch.randn(self.population_size, self.n_params)
            samples = self.mean.view(1, -1) + self.sigma * (z_samples @ L.T)
            z_values = z_samples  # Store for later adaptation
        except:
            # Fallback if covariance matrix is not positive definite
            print("Warning: Covariance matrix issue - using identity instead")
            self.C = torch.eye(self.n_params)
            z_values = torch.randn(self.population_size, self.n_params)
            samples = self.mean.view(1, -1) + self.sigma * z_values

        return samples, z_values
    
    def evaluate_population(self, population, dataloader, reward_func):
        """Evaluate fitness of each individual in population"""
        fitness_values = []
        energy_values = []
        original_params = self.get_flat_params()
        
        for i, params in enumerate(population):
            # Set model parameters
            self.set_flat_params(params)
            
            # Reset energy counter
            self.model.reset_energy_counter()
            self.model.reset_thermal_state()
            
            # Evaluate on training data
            total_reward = 0
            n_batches = 0
            
            for inputs, targets in dataloader:
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = F.mse_loss(outputs, targets)
                    reward = -loss.item()  # Negative loss as reward
                    total_reward += reward
                    n_batches += 1
                    
                # Check energy constraint
                if self.model.get_energy_consumed() > self.energy_constraint:
                    # Apply energy penalty
                    total_reward *= 0.5 * (self.energy_constraint / self.model.get_energy_consumed())
                    break
            
            # Calculate average reward
            avg_reward = total_reward / n_batches if n_batches > 0 else -float('inf')
            energy_used = self.model.get_energy_consumed()
            
            fitness_values.append(avg_reward)
            energy_values.append(energy_used)
        
        # Restore original parameters
        self.set_flat_params(original_params)
        
        return torch.tensor(fitness_values), torch.tensor(energy_values)
    
    def update_parameters(self, population, z_values, fitness, test_func):
        """Update parameters using CMA-ES update rules"""
        # Sort by fitness
        indices = torch.argsort(fitness, descending=True)
        sorted_z = z_values[indices]
        
        # Select top half of population
        mu = torch.tensor(self.population_size // 2)
        z_elite = sorted_z[:mu]
        
        # Compute weighted average
        weights = torch.log(mu + 0.5) - torch.log(torch.arange(mu) + 1)
        weights = weights / weights.sum()
        
        # Compute weighted mean of selected solutions
        z_weighted_avg = (weights.view(-1, 1) * z_elite).sum(0)
        
        # Compute new mean
        old_mean = self.mean.clone()
        self.mean = self.mean + self.learning_rate * self.sigma * (self.C @ z_weighted_avg)
        
        # Update evolution paths
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(self.c_sigma * (2 - self.c_sigma)) * z_weighted_avg
        self.p_c = (1 - self.c_c) * self.p_c + np.sqrt(self.c_c * (2 - self.c_c)) * (self.mean - old_mean) / self.sigma
        
        # Update covariance matrix
        rank_one = torch.outer(self.p_c, self.p_c)
        rank_mu = torch.zeros_like(self.C)
        for i in range(mu):
            rank_mu += weights[i] * torch.outer(z_elite[i], z_elite[i])
        
        self.C = (1 - self.c_1 - self.c_mu) * self.C + self.c_1 * rank_one + self.c_mu * rank_mu
        
        # Ensure symmetric positive definite
        self.C = (self.C + self.C.T) / 2
        
        # Add small regularization to diagonal
        self.C += torch.eye(self.n_params) * 1e-5
        
        # Update sigma (step size)
        if self.adapt_sigma:
            expected_norm = np.sqrt(self.n_params) * (1 - 1/(4*self.n_params) + 1/(21*self.n_params**2))
            sigma_scale = torch.norm(self.p_sigma) / expected_norm
            self.sigma *= np.exp((sigma_scale - 1) * self.c_sigma / 2)
            self.sigma = torch.clamp(torch.tensor(self.sigma), 0.001, 1.0).item()
        
        # Set model to best parameters (mean)
        self.set_flat_params(self.mean)
        
        # Evaluate test performance
        test_performance = test_func(self.model)
        
        # Store history
        self.fitness_history.append(fitness.max().item())
        self.sigma_history.append(self.sigma)
        
        return test_performance
    
    def train_epoch(self, dataloader, test_func):
        """Train for one epoch using CMA-ES"""
        self.model.train()
        
        # Sample population
        population, z_values = self.sample_population()
        
        # Evaluate population
        fitness, energy = self.evaluate_population(population, dataloader, None)
        
        # Update parameters
        test_loss = self.update_parameters(population, z_values, fitness, test_func)
        
        # Store energy history
        self.energy_history.append(energy.mean().item())
        
        return -fitness.max().item(), energy.mean().item(), test_loss


class TrustRegionPolicyOptimization:
    """
    Trust Region Policy Optimization (TRPO) adapted for photonic neural networks
    Focuses on making safe updates with minimal energy consumption
    """
    def __init__(self, model, max_kl=0.01, damping=0.1, cg_iters=10, energy_constraint=1000):
        self.model = model
        self.model.using_rl = True
        
        # TRPO parameters
        self.max_kl = max_kl
        self.damping = damping
        self.cg_iters = cg_iters
        self.energy_constraint = energy_constraint
        
        # Get parameter dimensions
        self.n_params = sum(p.numel() for p in model.parameters())
        
        # History for tracking
        self.fitness_history = []
        self.energy_history = []
        self.kl_history = []
        
    def get_flat_params(self):
        """Get flattened parameters from model"""
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])
        
    def set_flat_params(self, flat_params):
        """Set flattened parameters to model"""
        idx = 0
        for param in self.model.parameters():
            flat_size = param.numel()
            param.data.copy_(flat_params[idx:idx + flat_size].view(param.shape))
            idx += flat_size
            
    def conjugate_gradient(self, Avp_func, b, nsteps=10, residual_tol=1e-10):
        """
        Conjugate gradient algorithm
        Used to solve Ax = b where we only have access to A through A*v products
        """
        x = torch.zeros_like(b)
        r = b.clone()  # Residual
        p = b.clone()  # Search direction
        
        for i in range(nsteps):
            Avp = Avp_func(p)
            alpha = torch.dot(r, r) / (torch.dot(p, Avp) + 1e-8)
            x += alpha * p
            
            r_new = r - alpha * Avp
            if torch.norm(r_new) < residual_tol:
                break
                
            beta = torch.dot(r_new, r_new) / (torch.dot(r, r) + 1e-8)
            p = r_new + beta * p
            r = r_new
            
        return x
    
    def compute_fisher_vector_product(self, v, states, old_params):
        """
        Compute Fisher Vector Product (FVP): F*v
        This approximates the Hessian-vector product
        """
        # Restore old parameters
        states.requires_grad_(True)
        old_params_saved = self.get_flat_params()
        self.set_flat_params(old_params)
        
        # Forward pass to compute KL
        self.model.reset_energy_counter()
        
        self.model.train()
        states.requires_grad_(True)

        old_outputs = self.model(states, record_activations=True)
        old_activations = self.model.activation_traces
        
        # Set back current parameters
        self.set_flat_params(old_params_saved)
        
        # Forward pass with current parameters
        self.model.train()
        new_outputs = self.model(states, record_activations=True)
        new_activations = self.model.activation_traces
        
        # Compute KL between output activations (since we can't access all intermediate states)
        kl = torch.mean(self.compute_output_kl(old_activations[-1], new_activations[-1]))
        kl.requires_grad_(True)
        
        print(f"Old Activations Requires Grad: {old_activations[-1].requires_grad}")
        print(f"New Activations Requires Grad: {new_activations[-1].requires_grad}")
        print(f"KL Requires Grad: {kl.requires_grad}")

        # Compute gradient of KL wrt parameters
        grads = torch.autograd.grad(kl, self.model.parameters(), create_graph=True, allow_unused=True)
        grad_tensors = [g.view(-1) for g in grads if g is not None]
        if len(grad_tensors) == 0:
            raise RuntimeError("All gradients are None. Ensure KL divergence depends on model parameters.")
        flat_grad_kl = torch.cat(grad_tensors)
        
        # Compute product with v
        grad_kl_v = torch.dot(flat_grad_kl, v)
        
        # Compute Hessian-vector product
        hvp = torch.autograd.grad(grad_kl_v, self.model.parameters())
        flat_hvp = torch.cat([g.view(-1) for g in hvp])
        
        # Add damping
        return flat_hvp + self.damping * v
    
    def compute_output_kl(self, p, q):
        variance = 0.1  # Keep variance as a float
        p = p.clone().detach().requires_grad_(True)  # Ensure p requires grad
        q = q.clone().detach().requires_grad_(True)  # Ensure q requires grad
        kl_div = 0.5 * ((p - q)**2 / variance).sum(dim=1).mean()
        return kl_div
        
    def line_search(self, states, targets, old_loss, old_params, fullstep, expected_improve):
        """
        Backtracking line search
        Ensures the update improves the objective and satisfies the KL constraint
        """
        max_backtracks = 10
        accept_ratio = 0.1
        
        for i in range(max_backtracks):
            step_size = 0.5**i
            new_params = old_params + step_size * fullstep
            
            # Try new parameters
            self.set_flat_params(new_params)
            
            # Check performance
            self.model.reset_energy_counter()
            with torch.no_grad():
                new_outputs = self.model(states)
                new_loss = F.mse_loss(new_outputs, targets).item()
                
                # Also check KL constraint
                old_outputs = self.model(states, record_activations=True)
                old_act = self.model.activation_traces
                
                self.set_flat_params(old_params)
                self.model(states, record_activations=True)
                new_act = self.model.activation_traces
                
                kl = torch.mean(self.compute_output_kl(old_act[-1], new_act[-1])).item()
            
            # Calculate actual improvement
            improve = old_loss - new_loss
            
            # Accept if sufficient improvement and KL constraint satisfied
            if improve > 0 and improve > accept_ratio * expected_improve and kl < self.max_kl:
                return new_params, new_loss, kl
                
        # If no update is good, return old parameters
        return old_params, old_loss, 0.0
        
    def train_epoch(self, dataloader, test_func):
        """Train for one epoch using TRPO"""
        self.model.train()
        
        # Collect all data for batch update
        all_states = []
        all_targets = []
        
        for states, targets in dataloader:
            all_states.append(states)
            all_targets.append(targets)
            
            # Break if too much energy used (avoid processing entire dataset)
            if self.model.get_energy_consumed() > self.energy_constraint / 2:
                break
                
        if not all_states:
            return float('inf'), 0.0, float('inf')
            
        # Concatenate data
        states = torch.cat(all_states)
        targets = torch.cat(all_targets)
        
        # Save current parameters
        states.requires_grad_(True)  # Ensure states track gradients
        outputs = self.model(states)  # Ensure forward pass happens with tracked states
        self.model.zero_grad()  # Ensure no stale gradients affect computation
        old_params = self.get_flat_params()

        
        # Compute current loss
        self.model.reset_energy_counter()
        with torch.no_grad():
            outputs = self.model(states)
            old_loss = F.mse_loss(outputs, targets).item()
        
        # Compute loss gradient
        states.requires_grad_(True)
        outputs = self.model(states)
        loss = F.mse_loss(outputs, targets)
        self.model.zero_grad()
        loss.backward()
        
        # Get gradient
        policy_grad = torch.cat([p.grad.view(-1) for p in self.model.parameters()]).detach()
        
        # Define Fisher-vector product function
        Avp_func = lambda v: self.compute_fisher_vector_product(v, states, old_params)
        
        # Compute step direction using conjugate gradient
        step_dir = self.conjugate_gradient(Avp_func, -policy_grad, nsteps=self.cg_iters)
        
        # Compute step size
        shs = 0.5 * torch.dot(step_dir, Avp_func(step_dir))
        lm = torch.sqrt(2 * self.max_kl / (shs + 1e-8))
        fullstep = lm * step_dir
        
        # Expected improvement
        expected_improve = -torch.dot(policy_grad, fullstep)
        
        # Line search
        new_params, new_loss, kl = self.line_search(
            states, targets, old_loss, old_params, fullstep, expected_improve)
        
        # Set parameters to result of line search
        self.set_flat_params(new_params)
        
        # Check test performance
        test_loss = test_func(self.model)
        
        # Record energy used
        energy_used = self.model.get_energy_consumed()
        
        # Store history
        self.fitness_history.append(-new_loss)
        self.energy_history.append(energy_used)
        self.kl_history.append(kl)
        
        return new_loss, energy_used, test_loss


def generate_optical_dataset(n_samples=1000, noise=0.1):
    """
    Generate a synthetic dataset mimicking optical signal processing tasks
    Represents the kind of data a photonic neural network might process
    """
    # Input features representing optical signals
    X = torch.rand(n_samples, 8) * 2 - 1  # Between -1 and 1
    
    # Create target function with non-linear interactions and phase-like behaviors
    phases = torch.rand(n_samples, 3) * 2 * np.pi
    
    # Interference-like patterns
    y1 = torch.sin(X[:, 0] * X[:, 1] + phases[:, 0])
    y2 = torch.cos(X[:, 2] + X[:, 3] * phases[:, 1])
    y3 = 0.5 * torch.tanh(X[:, 4] * X[:, 5] + X[:, 6] * X[:, 7])
    
    # Combined output (representing spectrum or interference pattern)
    y = torch.stack([y1, y2, y3], dim=1)
    
    # Add measurement noise
    y = y + torch.randn_like(y) * noise
    
    # Split into train/validation/test
    train_X, train_y = X[:700], y[:700]
    val_X, val_y = X[700:850], y[700:850]
    test_X, test_y = X[850:], y[850:]
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    return train_loader, val_loader, test_loader

def evaluate_model(model, dataloader):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs, add_noise=False)
            loss = F.mse_loss(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def compare_training_methods(epochs=30, plot_results=True):
    """Compare different training methods for photonic neural networks"""
    # Generate dataset representing optical processing tasks
    train_loader, val_loader, test_loader = generate_optical_dataset()
    
    # Create a test function
    def test_func(model):
        return evaluate_model(model, test_loader)
    
    # Create models with identical architecture
    hidden_sizes = [20, 15]
    
    # Model for CMA-ES
    cma_model = PhotonicNeuralNetwork(
        input_size=8, 
        hidden_sizes=hidden_sizes, 
        output_size=3, 
        noise_level=0.02,
        crosstalk_factor=0.05
    )
    cma_trainer = CovarianceMatrixAdaptation(
        cma_model, 
        population_size=16, 
        sigma=0.1, 
        learning_rate=0.1,
        energy_constraint=500
    )
    
    # Model for TRPO
    trpo_model = PhotonicNeuralNetwork(
        input_size=8, 
        hidden_sizes=hidden_sizes, 
        output_size=3, 
        noise_level=0.02,
        crosstalk_factor=0.05
    )
    trpo_trainer = TrustRegionPolicyOptimization(
        trpo_model, 
        max_kl=0.01, 
        damping=0.1, 
        cg_iters=10,
        energy_constraint=500
    )
    
    # Model for traditional backpropagation (baseline)
    bp_model = PhotonicNeuralNetwork(
        input_size=8, 
        hidden_sizes=hidden_sizes, 
        output_size=3, 
        noise_level=0.02,
        crosstalk_factor=0.05
    )
    bp_optimizer = torch.optim.Adam(bp_model.parameters(), lr=0.01)
    
    # Training loop
    cma_train_losses = []
    cma_test_losses = []
    cma_energy = []
    
    trpo_train_losses = []
    trpo_test_losses = []
    trpo_energy = []
    
    bp_train_losses = []
    bp_test_losses = []
    bp_energy = []
    
    start_time = time.time()
    
    # Train for specified epochs
    for epoch in range(epochs):
        # -----------------------------------------------
        # Train CMA-ES model
        # -----------------------------------------------
        print(f"Epoch {epoch+1}/{epochs}")
        print("Training CMA-ES model...")
        cma_model.reset_energy_counter()
        cma_model.reset_thermal_state()
        train_loss, energy, test_loss = cma_trainer.train_epoch(train_loader, test_func)
        
        # Record metrics
        cma_train_losses.append(train_loss)
        cma_test_losses.append(test_loss)
        cma_energy.append(energy)
        
        print(f"  CMA-ES - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Energy: {energy:.2f}, Sigma: {cma_trainer.sigma:.4f}")
        
        # -----------------------------------------------
        # Train TRPO model
        # -----------------------------------------------
        print("Training TRPO model...")
        trpo_model.reset_energy_counter()
        trpo_model.reset_thermal_state()
        train_loss, energy, test_loss = trpo_trainer.train_epoch(train_loader, test_func)
        
        # Record metrics
        trpo_train_losses.append(train_loss)
        trpo_test_losses.append(test_loss)
        trpo_energy.append(energy)
        
        print(f"  TRPO - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Energy: {energy:.2f}, KL: {trpo_trainer.kl_history[-1]:.4f}")
        
        # -----------------------------------------------
        # Train BP model (baseline)
        # -----------------------------------------------
        print("Training BP model...")
        bp_model.train()
        bp_model.reset_energy_counter()
        bp_model.reset_thermal_state()
        epoch_loss = 0
        n_batches = 0
        
        for inputs, targets in train_loader:
            bp_optimizer.zero_grad()
            outputs = bp_model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            bp_optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        # Record metrics
        bp_train_losses.append(epoch_loss / n_batches)
        bp_test_losses.append(evaluate_model(bp_model, test_loader))
        bp_energy.append(bp_model.get_energy_consumed())
        
        print(f"  BP - Train Loss: {bp_train_losses[-1]:.4f}, Test Loss: {bp_test_losses[-1]:.4f}, Energy: {bp_energy[-1]:.2f}")
        print("-" * 50)
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    # Plot results
    if plot_results:
        plt.figure(figsize=(20, 15))
        
        # Training loss
        plt.subplot(3, 2, 1)
        plt.plot(cma_train_losses, label='CMA-ES')
        plt.plot(trpo_train_losses, label='TRPO')
        plt.plot(bp_train_losses, label='BP')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.title('Training Loss Comparison')
        
        # Test loss
        plt.subplot(3, 2, 2)
        plt.plot(cma_test_losses, label='CMA-ES')
        plt.plot(trpo_test_losses, label='TRPO')
        plt.plot(bp_test_losses, label='BP')
        plt.xlabel('Epoch')
        plt.ylabel('Test Loss')
        plt.legend()
        plt.title('Test Loss Comparison')
        
        # Energy consumption
        plt.subplot(3, 2, 3)
        plt.plot(cma_energy, label='CMA-ES')
        plt.plot(trpo_energy, label='TRPO')
        plt.plot(bp_energy, label='BP')
        plt.xlabel('Epoch')
        plt.ylabel('Energy Consumed')
        plt.legend()
        plt.title('Energy Consumption')
        
        # Loss vs Energy tradeoff
        plt.subplot(3, 2, 4)
        plt.scatter(np.cumsum(cma_energy), cma_test_losses, label='CMA-ES', alpha=0.7)
        plt.scatter(np.cumsum(trpo_energy), trpo_test_losses, label='TRPO', alpha=0.7)
        plt.scatter(np.cumsum(bp_energy), bp_test_losses, label='BP', alpha=0.7)
        plt.xlabel('Cumulative Energy')
        plt.ylabel('Test Loss')
        plt.legend()
        plt.title('Performance vs Energy Trade-off')
        
        # CMA-ES specific plot - sigma adaptation
        plt.subplot(3, 2, 5)
        plt.plot(cma_trainer.sigma_history)
        plt.xlabel('Epoch')
        plt.ylabel('Step Size (Sigma)')
        plt.title('CMA-ES Step Size Adaptation')
        
        # TRPO specific plot - KL divergence
        plt.subplot(3, 2, 6)
        plt.plot(trpo_trainer.kl_history)
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence')
        plt.title('TRPO Trust Region Constraint')
        
        plt.tight_layout()
        plt.savefig('photonic_training_comparison.png')
        plt.show()
    
    # Return final results
    return {
        'cma': {
            'model': cma_model,
            'final_loss': cma_test_losses[-1],
            'total_energy': sum(cma_energy)
        },
        'trpo': {
            'model': trpo_model,
            'final_loss': trpo_test_losses[-1],
            'total_energy': sum(trpo_energy)
        },
        'bp': {
            'model': bp_model,
            'final_loss': bp_test_losses[-1],
            'total_energy': sum(bp_energy)
        }
    }


def adaptive_stopping_criterion(energy_history, loss_history, window=5, threshold=0.01):
    """
    Adaptive early stopping criterion based on energy efficiency
    Stops training when the loss improvement per unit energy falls below threshold
    """
    if len(energy_history) < window + 1:
        return False
    
    # Calculate recent improvement in loss
    recent_loss_improve = loss_history[-window-1] - loss_history[-1]
    
    # Calculate energy spent in that period
    recent_energy = sum(energy_history[-window:])
    
    # Improvement per unit energy
    if recent_energy > 0:
        efficiency = recent_loss_improve / recent_energy
        return efficiency < threshold
    
    return False


def train_with_adaptive_energy_allocation(model, train_loader, val_loader, test_loader, 
                                          total_energy_budget=5000, max_epochs=50):
    """
    Energy-aware training that adaptively allocates energy based on improvements
    Automatically switches between training methods based on their efficiency
    """
    print("Starting adaptive energy-aware training...")
    
    # Create trainers
    cma_trainer = CovarianceMatrixAdaptation(
        model, 
        population_size=16, 
        sigma=0.1, 
        learning_rate=0.1,
        energy_constraint=total_energy_budget * 0.1  # Initial constraint per epoch
    )
    
    trpo_trainer = TrustRegionPolicyOptimization(
        model, 
        max_kl=0.01, 
        damping=0.1, 
        cg_iters=10,
        energy_constraint=total_energy_budget * 0.1
    )
    
    # History tracking
    energy_used = 0
    energy_history = []
    loss_history = []
    
    # Define test function
    def test_func(model):
        return evaluate_model(model, val_loader)
    
    current_method = "cma"  # Start with CMA-ES
    method_switch_count = 0
    
    # Use validation loss to determine efficiency
    best_val_loss = float('inf')
    
    # Train until energy budget is exhausted or max epochs reached
    epoch = 0
    while energy_used < total_energy_budget and epoch < max_epochs:
        print(f"Epoch {epoch+1}, Energy used: {energy_used:.2f}/{total_energy_budget:.2f}")
        print(f"Current method: {current_method}")
        
        model.reset_energy_counter()
        model.reset_thermal_state()
        
        # Train with current method
        if current_method == "cma":
            train_loss, epoch_energy, val_loss = cma_trainer.train_epoch(train_loader, test_func)
            # Adapt sigma based on progress
            if len(loss_history) > 0:
                improvement = (loss_history[-1] - val_loss) / loss_history[-1]
                cma_trainer.decay_sigma(improvement)
        else:  # TRPO
            train_loss, epoch_energy, val_loss = trpo_trainer.train_epoch(train_loader, test_func)
        
        # Update tracking
        energy_used += epoch_energy
        energy_history.append(epoch_energy)
        loss_history.append(val_loss)
        
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Epoch Energy: {epoch_energy:.2f}")
        
        # Check if we should switch methods
        if len(loss_history) > 5:
            # Calculate recent efficiency
            recent_improve = (loss_history[-5] - loss_history[-1]) / loss_history[-5]
            if recent_improve < 0.02:  # If improvement is small
                # Switch methods
                current_method = "trpo" if current_method == "cma" else "cma"
                method_switch_count += 1
                print(f"Switching to {current_method} due to low improvement")
        
        # Adjust energy constraints based on remaining budget
        remaining_budget = total_energy_budget - energy_used
        remaining_epochs = max_epochs - epoch - 1
        
        if remaining_epochs > 0:
            epoch_budget = remaining_budget / remaining_epochs
            cma_trainer.energy_constraint = epoch_budget * 1.2  # Allow some flexibility
            trpo_trainer.energy_constraint = epoch_budget * 1.2
        
        # Check if we've found a new best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            
        # Check adaptive stopping criterion
        if adaptive_stopping_criterion(energy_history, loss_history):
            print("Early stopping triggered: Low improvement per energy unit")
            break
            
        epoch += 1
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    test_loss = evaluate_model(model, test_loader)
    
    print("\nTraining completed:")
    print(f"Total Energy Used: {energy_used:.2f}/{total_energy_budget:.2f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Method switches: {method_switch_count}")
    
    return {
        'model': model,
        'test_loss': test_loss,
        'energy_used': energy_used,
        'energy_history': energy_history,
        'loss_history': loss_history
    }


def analyze_hardware_sensitivity(method="cma-es", noise_levels=[0.01, 0.05, 0.1], 
                                crosstalk_levels=[0.01, 0.05, 0.1]):
    """
    Analyze the sensitivity of training methods to hardware imperfections
    Tests performance under various noise and crosstalk scenarios
    """
    print("Analyzing hardware sensitivity...")
    
    # Generate dataset
    train_loader, val_loader, test_loader = generate_optical_dataset()
    
    # Results storage
    results = {}
    
    for noise in noise_levels:
        for crosstalk in crosstalk_levels:
            print(f"\nTesting with noise={noise}, crosstalk={crosstalk}")
            
            # Create model with specific imperfections
            model = PhotonicNeuralNetwork(
                input_size=8,
                hidden_sizes=[20, 15],
                output_size=3,
                noise_level=noise,
                crosstalk_factor=crosstalk
            )
            
            # Select training method
            if method == "cma-es":
                trainer = CovarianceMatrixAdaptation(
                    model, 
                    population_size=16, 
                    sigma=0.1, 
                    energy_constraint=300
                )
            else:  # TRPO
                trainer = TrustRegionPolicyOptimization(
                    model, 
                    max_kl=0.01, 
                    energy_constraint=300
                )
            
            # Define test function
            def test_func(model):
                return evaluate_model(model, test_loader)
            
            # Train for 10 epochs
            train_losses = []
            test_losses = []
            energy_usage = []
            
            for epoch in range(10):
                train_loss, energy, test_loss = trainer.train_epoch(train_loader, test_func)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                energy_usage.append(energy)
                
                print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
            
            # Store results
            results[(noise, crosstalk)] = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'energy_usage': energy_usage,
                'final_test_loss': test_losses[-1],
                'total_energy': sum(energy_usage)
            }
    
    # Plot heat map of final performance
    plt.figure(figsize=(12, 5))
    
    # Test loss heatmap
    plt.subplot(1, 2, 1)
    test_matrix = np.zeros((len(noise_levels), len(crosstalk_levels)))
    for i, noise in enumerate(noise_levels):
        for j, crosstalk in enumerate(crosstalk_levels):
            test_matrix[i, j] = results[(noise, crosstalk)]['final_test_loss']
    
    plt.imshow(test_matrix, cmap='viridis')
    plt.colorbar(label='Test Loss')
    plt.xticks(range(len(crosstalk_levels)), crosstalk_levels)
    plt.yticks(range(len(noise_levels)), noise_levels)
    plt.xlabel('Crosstalk Factor')
    plt.ylabel('Noise Level')
    plt.title(f'{method.upper()} Test Loss by Hardware Conditions')
    
    # Energy usage heatmap
    plt.subplot(1, 2, 2)
    energy_matrix = np.zeros((len(noise_levels), len(crosstalk_levels)))
    for i, noise in enumerate(noise_levels):
        for j, crosstalk in enumerate(crosstalk_levels):
            energy_matrix[i, j] = results[(noise, crosstalk)]['total_energy']
    
    plt.imshow(energy_matrix, cmap='plasma')
    plt.colorbar(label='Total Energy')
    plt.xticks(range(len(crosstalk_levels)), crosstalk_levels)
    plt.yticks(range(len(noise_levels)), noise_levels)
    plt.xlabel('Crosstalk Factor')
    plt.ylabel('Noise Level')
    plt.title(f'{method.upper()} Energy Usage by Hardware Conditions')
    
    plt.tight_layout()
    plt.savefig(f'{method}_hardware_sensitivity.png')
    plt.show()
    
    return results


# Main experiment function
def run_main_experiment():
    """Run the main experiment comparing different training methods"""
    print("Starting main experiment...")
    
    # Compare standard training methods
    results = compare_training_methods(epochs=20)
    
    # Create model for adaptive training
    model = PhotonicNeuralNetwork(
        input_size=8, 
        hidden_sizes=[20, 15], 
        output_size=3
    )
    
    # Run adaptive training
    train_loader, val_loader, test_loader = generate_optical_dataset()
    adaptive_results = train_with_adaptive_energy_allocation(
        model, train_loader, val_loader, test_loader, 
        total_energy_budget=3000, max_epochs=30
    )
    
    # Print final comparison
    print("\nFinal Results Comparison:")
    print(f"CMA-ES: Loss={results['cma']['final_loss']:.4f}, Energy={results['cma']['total_energy']:.2f}")
    print(f"TRPO: Loss={results['trpo']['final_loss']:.4f}, Energy={results['trpo']['total_energy']:.2f}")
    print(f"BP (Baseline): Loss={results['bp']['final_loss']:.4f}, Energy={results['bp']['total_energy']:.2f}")
    print(f"Adaptive: Loss={adaptive_results['test_loss']:.4f}, Energy={adaptive_results['energy_used']:.2f}")
    
    # Analyze sensitivity to hardware conditions
    sensitivity_results = analyze_hardware_sensitivity(method="cma-es")
    
    return {
        'standard_results': results,
        'adaptive_results': adaptive_results,
        'sensitivity_results': sensitivity_results
    }


if __name__ == "__main__":
    run_main_experiment()
