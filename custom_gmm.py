
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from datetime import datetime

class EnhancedGMM:
    """
    Enhanced Gaussian Mixture Model implementation with visualization and validation features
    """
    def __init__(self, n_components, max_iter=100, tol=1e-4, random_state=None, 
                 covariance_type='full', verbose=False):
        """
        Initialize the GMM with extended parameters
        
        Parameters:
        -----------
        n_components : int
            Number of mixture components
        max_iter : int
            Maximum number of EM iterations
        tol : float
            Convergence threshold
        random_state : int or None
            Random seed for reproducibility
        covariance_type : str
            Type of covariance matrix ('full', 'diagonal')
        verbose : bool
            Whether to print training progress
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.verbose = verbose
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_history = []
        
        if random_state is not None:
            np.random.seed(random_state)

    def _validate_input(self, X):
        """Validate input data"""
        if X.ndim != 2:
            raise ValueError("Input data must be 2D")
        if X.shape[0] < self.n_components:
            raise ValueError("Number of samples must be greater than number of components")
        return X

    def initialize_parameters(self, X):
        """Initialize model parameters"""
        X = self._validate_input(X)
        n_samples, n_features = X.shape
        
        # Initialize means using k-means-like approach
        random_idx = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[random_idx].copy()
        
        # Initialize covariances
        if self.covariance_type == 'full':
            base_cov = np.cov(X.T) + np.eye(n_features) * 1e-6  # Add small value for stability
            self.covs = np.array([base_cov for _ in range(self.n_components)])
        else:  # diagonal
            base_cov = np.diag(np.var(X, axis=0)) + 1e-6
            self.covs = np.array([base_cov for _ in range(self.n_components)])
            
        # Initialize mixing coefficients
        self.weights = np.ones(self.n_components) / self.n_components

    def e_step(self, X):
        """Expectation step: compute responsibilities"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            try:
                gaussian = multivariate_normal(mean=self.means[k], cov=self.covs[k])
                responsibilities[:, k] = self.weights[k] * gaussian.pdf(X)
            except:
                # Handle numerical instability
                responsibilities[:, k] = 1e-300
                
        # Normalize responsibilities
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        responsibilities = np.where(row_sums > 0, responsibilities / row_sums, 1/self.n_components)
        return responsibilities

    def m_step(self, X, responsibilities):
        """Maximization step: update parameters"""
        n_samples, n_features = X.shape
        Nk = responsibilities.sum(axis=0)
        
        # Update weights
        self.weights = Nk / n_samples
        
        # Update means
        self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            cov = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]
            if self.covariance_type == 'diagonal':
                cov = np.diag(np.diag(cov))
            self.covs[k] = cov + np.eye(n_features) * 1e-6  # Regularization

    def fit(self, X):
        """Fit the model using EM algorithm"""
        self.initialize_parameters(X)
        log_likelihood_old = None
        
        for iteration in range(self.max_iter):
            responsibilities = self.e_step(X)
            self.m_step(X, responsibilities)
            
            log_likelihood = self.compute_log_likelihood(X)
            self.log_likelihood_history.append(log_likelihood)
            self.n_iter_ = iteration + 1
            
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Log Likelihood = {log_likelihood:.4f}")
                
            if log_likelihood_old is not None:
                diff = abs(log_likelihood - log_likelihood_old)
                if diff < self.tol:
                    self.converged_ = True
                    break
            log_likelihood_old = log_likelihood
            
        return self

    def compute_log_likelihood(self, X):
        """Compute the log likelihood of the data"""
        n_samples = X.shape[0]
        likelihood = np.zeros(n_samples)
        for k in range(self.n_components):
            gaussian = multivariate_normal(mean=self.means[k], cov=self.covs[k])
            likelihood += self.weights[k] * gaussian.pdf(X)
        return np.sum(np.log(likelihood + 1e-300))  # Avoid log(0)

    def predict(self, X):
        """Predict cluster labels"""
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        """Predict posterior probabilities"""
        return self.e_step(X)

    def visualize(self, X, labels=None):
        """Visualize the results (2D data only)"""
        if X.shape[1] != 2:
            raise ValueError("Visualization only supported for 2D data")
            
        plt.figure(figsize=(10, 8))
        if labels is None:
            labels = self.predict(X)
            
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.scatter(self.means[:, 0], self.means[:, 1], c='red', marker='x', s=200, 
                   label='Cluster Centers')
        plt.colorbar(scatter)
        plt.title('GMM Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_convergence(self):
        """Plot log likelihood convergence"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.n_iter_), self.log_likelihood_history, 'b-', linewidth=2)
        plt.title('Log Likelihood Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood')
        plt.grid(True)
        plt.show()

# Example usage and demonstration
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 300
    X1 = np.random.normal(0, 1, (n_samples, 2))
    X2 = np.random.normal(5, 1.5, (n_samples, 2))
    X3 = np.random.normal(-3, 1.2, (n_samples, 2))
    X = np.vstack([X1, X2, X3])
    
    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit model
    start_time = datetime.now()
    gmm = EnhancedGMM(n_components=3, max_iter=200, tol=1e-4, random_state=42, 
                     covariance_type='full', verbose=True)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    
    # Print results
    print(f"\nTraining completed in {datetime.now() - start_time}")
    print(f"Converged: {gmm.converged_}")
    print(f"Number of iterations: {gmm.n_iter_}")
    print(f"Final log likelihood: {gmm.log_likelihood_history[-1]:.4f}")
    print(f"Cluster sizes: {np.bincount(labels)}")
    
    # Visualize results
    gmm.visualize(X_scaled, labels)
    gmm.plot_convergence()