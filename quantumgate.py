import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
import qutip as qt
from tqdm import tqdm

class AdaptiveGateCalibration:
    """
    Framework for sample-efficient adaptive quantum gate calibration using
    Bayesian optimization and optimal control techniques.
    """
    def __init__(self, dim=2, target_gate=None, noise_level=0.01, seed=42):
        """
        Initialize the adaptive gate calibration framework.
        
        Parameters:
        -----------
        dim : int
            Dimension of the quantum system (default: 2 for a qubit)
        target_gate : np.ndarray
            Target gate unitary matrix
        noise_level : float
            Simulated experimental noise level
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed)
        self.dim = dim
        
        # Default target gate: X gate (NOT gate)
        if target_gate is None:
            self.target_gate = qt.sigmax().full()
        else:
            self.target_gate = target_gate
            
        self.noise_level = noise_level
        
        # Parameter space for control pulses
        self.param_bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [0.1, 3.0], [0.1, 3.0]])
        self.param_names = ['amplitude_x', 'amplitude_y', 'duration_x', 'duration_y']
        
        # System Hamiltonian components
        self.H0 = qt.sigmaz() * 2 * np.pi  # Drift Hamiltonian
        self.Hx = qt.sigmax() * 2 * np.pi  # Control Hamiltonian in x
        self.Hy = qt.sigmay() * 2 * np.pi  # Control Hamiltonian in y
        
        # Gate fidelity evaluation history
        self.param_history = []
        self.fidelity_history = []
        
        # Bayesian optimization setup
        self.gp = None
        self.init_gp()
        
    def init_gp(self):
        """Initialize the Gaussian Process Regressor for Bayesian optimization."""
        # Change in the init_gp method:
        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * Matern(length_scale=[0.5, 0.5, 0.5, 0.5], nu=2.5, length_scale_bounds=(1e-2, 10.0))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.noise_level**2,
            normalize_y=True,
            n_restarts_optimizer=10
        )
    
    def pulse_to_unitary(self, params):
        """
        Convert control pulse parameters to a unitary gate.
        
        Parameters:
        -----------
        params : np.ndarray
            Control pulse parameters [amplitude_x, amplitude_y, duration_x, duration_y]
            
        Returns:
        --------
        U : np.ndarray
            Resulting unitary matrix
        """
        amp_x, amp_y, duration_x, duration_y = params
        
        # Create time-dependent Hamiltonian
        def h_t(t, args):
            H = self.H0.copy()
            
            # X pulse
            if t < duration_x:
                H += amp_x * self.Hx
                
            # Y pulse
            if duration_x <= t < (duration_x + duration_y):
                H += amp_y * self.Hy
                
            return H
        
        # Add to the pulse_to_unitary method:
        def normalize_params(self, params):
            normalized = np.zeros_like(params)
            for i in range(len(params)):
                normalized[i] = (params[i] - self.param_bounds[i, 0]) / (self.param_bounds[i, 1] - self.param_bounds[i, 0])
            return normalized
        # Solve time evolution
        tlist = np.linspace(0, duration_x + duration_y, 100)
        result = qt.sesolve(h_t, qt.qeye(self.dim), tlist)
        
        # Extract final unitary
        U = result.states[-1].full()
        return U
    
    def compute_gate_fidelity(self, params, n_measurements=100):
        """
        Compute the gate fidelity with respect to the target gate.
        Simulates experimental noise and measurement overhead.
        
        Parameters:
        -----------
        params : np.ndarray
            Control pulse parameters
        n_measurements : int
            Number of simulated measurements for fidelity estimation
            
        Returns:
        --------
        fidelity : float
            Estimated gate fidelity
        """
        U = self.pulse_to_unitary(params)
        
        # Calculate true process fidelity
        Utarget_dag = self.target_gate.conj().T
        true_fidelity = np.abs(np.trace(Utarget_dag @ U) / self.dim)**2
        
        # Simulate experimental noise based on number of measurements
        noise = np.random.normal(0, self.noise_level / np.sqrt(n_measurements))
        measured_fidelity = max(0, min(1, true_fidelity + noise))
        
        return measured_fidelity
    
    def acquisition_function(self, params, method='ucb', kappa=2.0):
        """
        Acquisition function for Bayesian optimization.
        
        Parameters:
        -----------
        params : np.ndarray
            Control pulse parameters to evaluate
        method : str
            Acquisition function type ('ucb', 'ei', 'pi')
        kappa : float
            Exploration-exploitation trade-off parameter
            
        Returns:
        --------
        acq_value : float
            Acquisition function value
        """
        params = params.reshape(1, -1)
        
        # Compute mean and standard deviation
        mu, sigma = self.gp.predict(params, return_std=True)
        
        if method == 'ucb':  # Upper Confidence Bound
            return mu + kappa * sigma
        elif method == 'ei':  # Expected Improvement
            best_f = np.max(self.fidelity_history) if self.fidelity_history else 0
            imp = mu - best_f
            Z = imp / (sigma + 1e-6)
            return imp * (0.5 * (1 + np.math.erf(Z / np.sqrt(2))))
        elif method == 'pi':  # Probability of Improvement
            best_f = np.max(self.fidelity_history) if self.fidelity_history else 0
            Z = (mu - best_f) / (sigma + 1e-6)
            return 0.5 * (1 + np.math.erf(Z / np.sqrt(2)))
        else:
            raise ValueError(f"Unknown acquisition function: {method}")
    
    def next_point_to_sample(self, method='ucb', n_restarts=5):
        """
        Determine the next point to sample based on the acquisition function.
        
        Parameters:
        -----------
        method : str
            Acquisition function type
        n_restarts : int
            Number of random restarts for optimization
            
        Returns:
        --------
        best_params : np.ndarray
            Next parameters to sample
        """
        # Define the objective function to minimize
        def objective(params):
            return -self.acquisition_function(params, method=method)
        
        # Run optimization from multiple starting points
        best_params = None
        best_acq = -np.inf
        
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.random.uniform(
                self.param_bounds[:, 0],
                self.param_bounds[:, 1]
            )
            
            # Optimize acquisition function
            res = minimize(
                objective,
                x0,
                bounds=self.param_bounds,
                method='L-BFGS-B'
            )
            
            if -res.fun > best_acq:
                best_acq = -res.fun
                best_params = res.x
                
        return best_params
    
    def update_model(self, params, fidelity):
        """
        Update the Gaussian Process model with new data.
        
        Parameters:
        -----------
        params : np.ndarray
            Control pulse parameters
        fidelity : float
            Measured fidelity value
        """
        self.param_history.append(params)
        self.fidelity_history.append(fidelity)
        
        X = np.array(self.param_history)
        y = np.array(self.fidelity_history)
        
        # Update the GP model
        self.gp.fit(X, y)
    
    def run_optimization(self, n_iterations=20, initial_measurements=5, acq_method='ucb'):
        """
        Run the full Bayesian optimization loop.
        
        Parameters:
        -----------
        n_iterations : int
            Number of optimization iterations
        initial_measurements : int
            Number of initial random measurements
        acq_method : str
            Acquisition function type
            
        Returns:
        --------
        best_params : np.ndarray
            Optimal control parameters
        best_fidelity : float
            Best achieved fidelity
        """
        # Initial random sampling
        for _ in range(initial_measurements):
            params = np.random.uniform(
                self.param_bounds[:, 0],
                self.param_bounds[:, 1]
            )
            fidelity = self.compute_gate_fidelity(params)
            self.update_model(params, fidelity)
            
        # Bayesian optimization loop
        pbar = tqdm(range(n_iterations), desc="Optimizing gate parameters")
        for _ in pbar:
            next_params = self.next_point_to_sample(method=acq_method, kappa=3.0)
            next_fidelity = self.compute_gate_fidelity(next_params)
            self.update_model(next_params, next_fidelity)
            
            best_idx = np.argmax(self.fidelity_history)
            best_fid = self.fidelity_history[best_idx]
            pbar.set_postfix({"best_fidelity": f"{best_fid:.6f}"})
            
        # Return best parameters and fidelity
        best_idx = np.argmax(self.fidelity_history)
        best_params = self.param_history[best_idx]
        best_fidelity = self.fidelity_history[best_idx]
        
        return best_params, best_fidelity
    
    def plot_optimization_results(self):
        """Plot the optimization results including fidelity improvement over iterations."""
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Fidelity vs Iteration
        plt.subplot(2, 2, 1)
        plt.plot(range(len(self.fidelity_history)), self.fidelity_history, 'o-', color='blue')
        plt.axhline(y=1.0, linestyle='--', color='red')
        plt.title('Gate Fidelity vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Fidelity')
        plt.grid(True)
        
        # Plot 2: Parameter trajectories
        plt.subplot(2, 2, 2)
        params_array = np.array(self.param_history)
        for i, name in enumerate(self.param_names):
            plt.plot(range(len(params_array)), params_array[:, i], 'o-', label=name)
        plt.title('Parameter Values vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.legend()
        plt.grid(True)
        
        # If we have enough data, create prediction surface plots for 2D slices
        if len(self.param_history) >= 10:
            best_idx = np.argmax(self.fidelity_history)
            best_params = self.param_history[best_idx]
            
            # Plot 3: Predicted fidelity surface (amplitude_x vs amplitude_y)
            plt.subplot(2, 2, 3)
            xx, yy = np.meshgrid(
                np.linspace(self.param_bounds[0, 0], self.param_bounds[0, 1], 20),
                np.linspace(self.param_bounds[1, 0], self.param_bounds[1, 1], 20)
            )
            
            # Fix other parameters at their optimal values
            fixed_params = np.tile(best_params, (xx.size, 1))
            fixed_params[:, 0] = xx.ravel()
            fixed_params[:, 1] = yy.ravel()
            
            # Predict fidelity
            predicted_fidelity = self.gp.predict(fixed_params)
            predicted_fidelity = predicted_fidelity.reshape(xx.shape)
            
            # Plot contour
            plt.contourf(xx, yy, predicted_fidelity, levels=50, cmap='viridis')
            plt.colorbar(label='Predicted Fidelity')
            plt.plot(best_params[0], best_params[1], 'ro', markersize=10, label='Optimum')
            plt.title('Predicted Fidelity (amplitude_x vs amplitude_y)')
            plt.xlabel('amplitude_x')
            plt.ylabel('amplitude_y')
            plt.legend()
            
            # Plot 4: Predicted fidelity surface (duration_x vs duration_y)
            plt.subplot(2, 2, 4)
            xx, yy = np.meshgrid(
                np.linspace(self.param_bounds[2, 0], self.param_bounds[2, 1], 20),
                np.linspace(self.param_bounds[3, 0], self.param_bounds[3, 1], 20)
            )
            
            # Fix other parameters at their optimal values
            fixed_params = np.tile(best_params, (xx.size, 1))
            fixed_params[:, 2] = xx.ravel()
            fixed_params[:, 3] = yy.ravel()
            
            # Predict fidelity
            predicted_fidelity = self.gp.predict(fixed_params)
            predicted_fidelity = predicted_fidelity.reshape(xx.shape)
            
            # Plot contour
            plt.contourf(xx, yy, predicted_fidelity, levels=50, cmap='viridis')
            plt.colorbar(label='Predicted Fidelity')
            plt.plot(best_params[2], best_params[3], 'ro', markersize=10, label='Optimum')
            plt.title('Predicted Fidelity (duration_x vs duration_y)')
            plt.xlabel('duration_x')
            plt.ylabel('duration_y')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def compare_measurement_efficiency(self, methods=['random', 'grid', 'bayesian'], 
                                       n_iterations=20, n_runs=5):
        """
        Compare the efficiency of different optimization methods.
        
        Parameters:
        -----------
        methods : list
            List of methods to compare
        n_iterations : int
            Number of iterations for each method
        n_runs : int
            Number of runs for statistical comparison
            
        Returns:
        --------
        results : dict
            Dictionary of results for each method
        """
        results = {method: {'fidelities': [], 'params': []} for method in methods}
        
        for method in methods:
            print(f"Testing method: {method}")
            
            for run in range(n_runs):
                print(f"  Run {run+1}/{n_runs}")
                
                if method == 'random':
                    # Random search
                    best_fidelity = 0
                    best_params = None
                    
                    for _ in tqdm(range(n_iterations), desc="Random search"):
                        params = np.random.uniform(
                            self.param_bounds[:, 0],
                            self.param_bounds[:, 1]
                        )
                        fidelity = self.compute_gate_fidelity(params)
                        
                        if fidelity > best_fidelity:
                            best_fidelity = fidelity
                            best_params = params
                    
                elif method == 'grid':
                    # Grid search (simplified for 4D)
                    points_per_dim = max(2, int(n_iterations**(1/4)))
                    
                    best_fidelity = 0
                    best_params = None
                    
                    # Create grid points along each dimension
                    grid_points = []
                    for i in range(4):
                        grid_points.append(np.linspace(
                            self.param_bounds[i, 0],
                            self.param_bounds[i, 1],
                            points_per_dim
                        ))
                    
                    # Sample from grid
                    total_points = min(n_iterations, points_per_dim**4)
                    indices = np.random.choice(points_per_dim**4, total_points, replace=False)
                    
                    for idx in tqdm(indices, desc="Grid search"):
                        # Convert flat index to multi-dimensional indices
                        idx_0 = idx // (points_per_dim**3) % points_per_dim
                        idx_1 = idx // (points_per_dim**2) % points_per_dim
                        idx_2 = idx // points_per_dim % points_per_dim
                        idx_3 = idx % points_per_dim
                        
                        params = np.array([
                            grid_points[0][idx_0],
                            grid_points[1][idx_1],
                            grid_points[2][idx_2],
                            grid_points[3][idx_3]
                        ])
                        
                        fidelity = self.compute_gate_fidelity(params)
                        
                        if fidelity > best_fidelity:
                            best_fidelity = fidelity
                            best_params = params
                
                elif method == 'bayesian':
                    # Reset for this run
                    self.param_history = []
                    self.fidelity_history = []
                    self.init_gp()
                    
                    # Run Bayesian optimization
                    best_params, best_fidelity = self.run_optimization(
                        n_iterations=n_iterations-5,  # Account for initial samples
                        initial_measurements=5,
                        acq_method='ucb'
                    )
                
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Store results
                results[method]['fidelities'].append(best_fidelity)
                results[method]['params'].append(best_params)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        
        for method in methods:
            fids = results[method]['fidelities']
            plt.bar(method, np.mean(fids), yerr=np.std(fids), capsize=5, 
                   alpha=0.7, label=method)
        
        plt.ylabel('Best Fidelity Achieved')
        plt.title('Comparison of Optimization Methods')
        plt.ylim([0.8, 1.0])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return results


class QuantumCharacterizationProtocols:
    """
    Implementation of various quantum characterization protocols for efficient gate calibration.
    """
    def __init__(self, dim=2):
        """
        Initialize quantum characterization protocols.
        
        Parameters:
        -----------
        dim : int
            Dimension of the quantum system
        """
        self.dim = dim
    
    def randomized_benchmarking(self, gate_sequence, n_shots=100, noise_level=0.01):
        """
        Simulate randomized benchmarking protocol.
        
        Parameters:
        -----------
        gate_sequence : list
            List of gate unitaries
        n_shots : int
            Number of measurements
        noise_level : float
            Simulated noise level
            
        Returns:
        --------
        fidelity : float
            Estimated gate fidelity
        """
        # Initialize with identity
        current_state = qt.qeye(self.dim)
        
        # Apply sequence of gates
        for gate in gate_sequence:
            current_state = gate @ current_state
        
        # Calculate ideal survival probability
        target_state = qt.qeye(self.dim)
        ideal_prob = np.abs(np.trace(target_state.dag() @ current_state) / self.dim)**2
        
        # Add measurement noise
        noise = np.random.normal(0, noise_level / np.sqrt(n_shots))
        measured_prob = max(0, min(1, ideal_prob + noise))
        
        return measured_prob
    
    def gate_set_tomography(self, gate, n_measurements=100, noise_level=0.01):
        """
        Simulate simplified gate set tomography.
        
        Parameters:
        -----------
        gate : np.ndarray
            Gate unitary matrix
        n_measurements : int
            Number of measurements per configuration
        noise_level : float
            Simulated noise level
            
        Returns:
        --------
        estimated_gate : np.ndarray
            Estimated gate unitary
        """
        # Define measurement bases for a qubit
        if self.dim == 2:
            bases = [
                (qt.basis(2, 0), qt.basis(2, 0)),  # |0⟩⟨0|
                (qt.basis(2, 1), qt.basis(2, 1)),  # |1⟩⟨1|
                (qt.snot() * qt.basis(2, 0), qt.snot() * qt.basis(2, 0)),  # |+⟩⟨+|
                (qt.snot() * qt.basis(2, 1), qt.snot() * qt.basis(2, 1)),  # |-⟩⟨-|
                ((qt.snot() * qt.sigmay()) * qt.basis(2, 0), 
                 (qt.snot() * qt.sigmay()) * qt.basis(2, 0))  # |+i⟩⟨+i|
            ]
        else:
            # For higher dimensions, we would need more bases
            bases = [(qt.basis(self.dim, i), qt.basis(self.dim, j)) 
                    for i in range(self.dim) for j in range(self.dim)]
        
        # Simulate measurements
        measurements = []
        
        for in_state, out_proj in bases:
            # True probability
            evolved = gate * in_state
            true_prob = (evolved.dag() * out_proj * evolved).tr().real
            
            # Add measurement noise
            noise = np.random.normal(0, noise_level / np.sqrt(n_measurements))
            measured_prob = max(0, min(1, true_prob + noise))
            
            measurements.append(measured_prob)
        
        # In real GST, a maximum likelihood estimation would reconstruct the gate
        # Here we'll simulate by adding noise to the true gate matrix
        process_noise = np.random.normal(0, noise_level / np.sqrt(len(measurements)), 
                                       size=(self.dim, self.dim)) + \
                      1j * np.random.normal(0, noise_level / np.sqrt(len(measurements)), 
                                          size=(self.dim, self.dim))
        
        estimated_gate = gate + process_noise
        
        # Ensure the result is unitary (approximate)
        u, s, vh = np.linalg.svd(estimated_gate)
        estimated_gate = u @ vh
        
        return estimated_gate
        
    def direct_fidelity_estimation(self, actual_gate, target_gate, n_samples=100, noise_level=0.01):
        """
        Simulate direct fidelity estimation protocol.
        
        Parameters:
        -----------
        actual_gate : np.ndarray
            Actual gate unitary
        target_gate : np.ndarray
            Target gate unitary
        n_samples : int
            Number of measurements
        noise_level : float
            Simulated noise level
            
        Returns:
        --------
        fidelity : float
            Estimated gate fidelity
        """
        # True fidelity
        target_dag = target_gate.conj().T
        true_fidelity = np.abs(np.trace(target_dag @ actual_gate) / self.dim)**2
        
        # Simulate sampling error
        noise = np.random.normal(0, noise_level / np.sqrt(n_samples))
        estimated_fidelity = max(0, min(1, true_fidelity + noise))
        
        return estimated_fidelity


def demo_gate_calibration():
    """Run a demonstration of the adaptive gate calibration framework."""
    print("Initializing Adaptive Gate Calibration Framework...")
    
    # Create X gate calibration task
    calibrator = AdaptiveGateCalibration(dim=2)
    
    print("Running Bayesian optimization for gate calibration...")
    best_params, best_fidelity = calibrator.run_optimization(n_iterations=20)
    
    print(f"Best parameters found: {best_params}")
    print(f"Best fidelity achieved: {best_fidelity:.6f}")
    
    # Plot results
    calibrator.plot_optimization_results()
    
    # Compare with other methods
    print("Comparing optimization methods...")
    results = calibrator.compare_measurement_efficiency(
        methods=['random', 'grid', 'bayesian'], 
        n_iterations=25,
        n_runs=3
    )
    
    # Show best parameters for each method
    for method, data in results.items():
        avg_fidelity = np.mean(data['fidelities'])
        std_fidelity = np.std(data['fidelities'])
        print(f"{method}: {avg_fidelity:.6f} ± {std_fidelity:.6f}")
    
    return calibrator, results


if __name__ == "__main__":
    # Run demonstration
    calibrator, results = demo_gate_calibration()
    
    # Demonstrate quantum characterization protocols
    print("\nDemonstrating Quantum Characterization Protocols...")
    qcp = QuantumCharacterizationProtocols(dim=2)
    
    # Create a sample gate with some error
    true_gate = qt.sigmax().full()  # X gate
    noisy_gate = qt.rx(np.pi * 0.95).full()  # Slightly off X gate
    
    # Estimate fidelity using different methods and measurement budgets
    print("\nFidelity estimation with different measurement budgets:")
    for n_shots in [10, 100, 1000]:
        # Direct Fidelity Estimation
        dfe_fidelity = qcp.direct_fidelity_estimation(noisy_gate, true_gate, n_samples=n_shots)
        
        # Randomized Benchmarking (simplified)
        rb_seq = [noisy_gate] * 10
        rb_fidelity = qcp.randomized_benchmarking(rb_seq, n_shots=n_shots)
        
        # Gate Set Tomography (simplified)
        est_gate = qcp.gate_set_tomography(noisy_gate, n_measurements=n_shots)
        gst_fidelity = np.abs(np.trace(true_gate.conj().T @ est_gate) / 2)**2
        
        print(f"Shots: {n_shots}")
        print(f"  Direct Fidelity Estimation: {dfe_fidelity:.6f}")
        print(f"  Randomized Benchmarking: {rb_fidelity:.6f}")
        print(f"  Gate Set Tomography: {gst_fidelity:.6f}")
