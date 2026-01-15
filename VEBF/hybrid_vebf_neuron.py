import numpy as np
from VEBF.vebf_neuron import VEBFNeuron

class HybridVEBFNeuron(VEBFNeuron):
    def __init__(self, center, label, n_dim, a_init, n_init=1):
        super().__init__(center, label, n_dim, a_init, n_init)
        # Initialize inv_cov_matrix
        self.inv_cov_matrix = np.eye(n_dim)

    def update_eigen_properties_hybrid(self, eigenvectors):
        """
        Updates eigenvectors with provided values (from LDA or PCA externally computed).
        Also updates the inverse covariance matrix for Mahalanobis distance.
        """
        self.eigenvectors = eigenvectors
        
        # Update inverse covariance matrix
        # Add regularization to avoid singular matrix
        reg = 1e-6 * np.eye(self.n_dim)
        try:
            self.inv_cov_matrix = np.linalg.inv(self.cov_matrix + reg)
        except np.linalg.LinAlgError:
            self.inv_cov_matrix = np.eye(self.n_dim)

    def calculate_mahalanobis(self, x):
        """
        Calculates Mahalanobis distance between x and neuron center.
        """
        diff = x - self.center
        # d = sqrt(diff.T * inv_cov * diff)
        dist_sq = diff.T @ self.inv_cov_matrix @ diff
        return np.sqrt(dist_sq)

    def update_params(self, x_new, center_new, cov_new, n0=2, eigenvectors=None):
        """
        Update parameters with option to provide eigenvectors directly.
        """
        center_old = self.center.copy()
        
        self.center = center_new
        self.cov_matrix = cov_new
        self.n += 1
        
        if eigenvectors is not None:
             self.update_eigen_properties_hybrid(eigenvectors)
        else:
             # Fallback to standard PCA if no eigenvectors provided (should not happen in Hybrid Algorithm context usually, but good for safety)
             self.update_eigen_properties()
             reg = 1e-6 * np.eye(self.n_dim)
             try:
                self.inv_cov_matrix = np.linalg.inv(self.cov_matrix + reg)
             except np.linalg.LinAlgError:
                self.inv_cov_matrix = np.eye(self.n_dim)
        
        # Update 'a' (width vector)
        if self.n > n0:
            shift_vecter = self.center - center_old
            # Ensure eigenvectors are present
            projected_shift = np.abs(np.dot(shift_vecter, self.eigenvectors))
            self.a += projected_shift
