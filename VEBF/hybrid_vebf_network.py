import numpy as np
from VEBF.vebf_network import VEBFNetwork
from VEBF.hybrid_vebf_neuron import HybridVEBFNeuron
from VEBF.utils import calculate_initial_width_hybrid

class HybridVEBFNetwork(VEBFNetwork):
    def __init__(self, X_train, n_classes, delta=1.0, n0=5, theta=-0.5):
        # Initialize with parent logic (to set n0, theta)
        # Passing delta=1.0 as dummy, we will overwrite a0.
        super().__init__(X_train, delta, n0, theta)
        
        self.n_classes = n_classes
        
        # Overwrite a0 with Algorithm 1 (Hybrid)
        self.a0 = calculate_initial_width_hybrid(X_train, n_classes)
        print(f"Hybrid calculated initial a0: {self.a0}")
        
        # Reset neurons (parent init might have set empty list, which is fine)
        self.neurons = [] 
        
        # Global LDA Statistics
        self.n_dim = X_train.shape[1]
        self.class_means = {} # label -> mean vector
        self.class_counts = {} # label -> count (n_q)
        self.global_mean = np.zeros(self.n_dim)
        self.global_count = 0
        self.S_W = np.zeros((self.n_dim, self.n_dim))
        
    def create_neuron(self, x, t):
        # Override to use HybridVEBFNeuron and self.a0
        neuron = HybridVEBFNeuron(center=x, label=t, n_dim=len(x), a_init=self.a0.copy())
        self.neurons.append(neuron)
        
    def update_global_stats(self, x, t):
        # Update global mean
        # mu_new = (N * mu + x) / (N + 1)
        self.global_mean = (self.global_count * self.global_mean + x) / (self.global_count + 1)
        self.global_count += 1
        
        # Update class mean and S_W
        if t not in self.class_means:
            self.class_means[t] = np.zeros(self.n_dim)
            self.class_counts[t] = 0
            
        old_class_mean = self.class_means[t].copy()
        n_t = self.class_counts[t]
        
        # Update class mean
        self.class_means[t] = (n_t * self.class_means[t] + x) / (n_t + 1)
        self.class_counts[t] += 1
        
        # Update S_W
        # S_W_new = S_W_old + (n_q / (n_q + 1)) * (x - mu_q_old) * (x - mu_q_old)^T
        diff = (x - old_class_mean).reshape(-1, 1)
        term = (1.0 / (n_t + 1)) * n_t * np.dot(diff, diff.T) if n_t > 0 else np.zeros((self.n_dim, self.n_dim))
        
        self.S_W += term
        
    def compute_lda_eigenvectors(self):
        # Calculate S_B
        S_B = np.zeros((self.n_dim, self.n_dim))
        for t, mean in self.class_means.items():
            n_t = self.class_counts[t]
            diff = (mean - self.global_mean).reshape(-1, 1)
            S_B += n_t * np.dot(diff, diff.T)
            
        # Check S_W nonsingular
        if self.S_W.size == 0: return None
        
        try:
            # Check for singularity
            if np.linalg.cond(self.S_W) > 1e10: # Ill-conditioned
                return None
                
            inv_S_W = np.linalg.inv(self.S_W)
            matrix_d = np.dot(inv_S_W, S_B)
            
            # Solve eigenvalue problem
            eigenvalues, eigenvectors = np.linalg.eig(matrix_d)
            
            # Use real part (should be real for symmetric generalized eigenproblem, 
            # though inv(A)*B is not symmetric, eigenvalues are real if A, B symmetric positive definite)
            eigenvectors = np.real(eigenvectors)
            eigenvalues = np.real(eigenvalues)
            
            # Sort by eigenvalue descending
            idx = eigenvalues.argsort()[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            return eigenvectors
            
        except np.linalg.LinAlgError:
            return None

    def learn_one_pass(self, x, t):
        # 1. Update Global Stats (Step 3.3 implies using updated stats)
        # Note: Paper says "3.3 ... if S_W is nonsingular". S_W is accumulated.
        self.update_global_stats(x, t)
        
        # 2. Find closest hidden neuron (same class)
        closest_neuron = None
        min_dist = float('inf')
        
        candidate_neurons = [n for n in self.neurons if n.label == t]
        
        if not candidate_neurons:
            self.create_neuron(x, t)
            return

        # Measured by Mahalanobis distance (Step 3.1)
        for neuron in candidate_neurons:
            dist = neuron.calculate_mahalanobis(x)
            if dist < min_dist:
                min_dist = dist
                closest_neuron = neuron
        
        if closest_neuron is None:
             self.create_neuron(x, t)
             return

        # 3.2 Compute new center and covariance
        center_new, cov_new = closest_neuron.calculate_provisional_params(x)
        
        # 3.3 Compute eigenvectors (LDA or PCA)
        lda_vectors = self.compute_lda_eigenvectors()
        
        if lda_vectors is not None:
             eigenvectors_new = lda_vectors
        else:
             # Fallback to PCA on the neuron's new covariance
             evals, evecs = np.linalg.eigh(cov_new)
             # eigh returns eigenvalues in ascending order
             eigenvectors_new = evecs[:, ::-1]
        
        # 3.4 Check Psi condition
        psi_new = closest_neuron.calculate_psi(x, center=center_new, eigenvectors=eigenvectors_new)
        
        if psi_new <= 0:
             # Update parameters
             closest_neuron.update_params(x, center_new, cov_new, n0=self.n0, eigenvectors=eigenvectors_new)
        else:
             # Create new neuron
             self.create_neuron(x, t)
