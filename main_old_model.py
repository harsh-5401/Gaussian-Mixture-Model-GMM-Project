# import numpy as np
# from scipy.stats import multivariate_normal

# class CustomGMM:
#     def __init__(self, n_components, max_iter=100, tol=1e-4, random_state=None):
#         self.n_components = n_components
#         self.max_iter = max_iter
#         self.tol = tol
#         self.random_state = random_state
#         if random_state is not None:
#             np.random.seed(random_state)

#     def initialize_parameters(self, X):
#         n_samples, n_features = X.shape
#         # Randomly initialize means
#         self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
#         # Initialize covariances as identity matrices
#         self.covs = np.array([np.cov(X.T) for _ in range(self.n_components)])
#         # Initialize equal weights
#         self.weights = np.ones(self.n_components) / self.n_components

#     def e_step(self, X):
#         n_samples = X.shape[0]
#         responsibilities = np.zeros((n_samples, self.n_components))
        
#         # Compute responsibilities (posterior probabilities)
#         for k in range(self.n_components):
#             gaussian = multivariate_normal(mean=self.means[k], cov=self.covs[k])
#             responsibilities[:, k] = self.weights[k] * gaussian.pdf(X)
        
#         # Normalize responsibilities
#         responsibilities /= responsibilities.sum(axis=1, keepdims=True)
#         return responsibilities

#     def m_step(self, X, responsibilities):
#         n_samples = X.shape[0]
        
#         # Update weights
#         Nk = responsibilities.sum(axis=0)
#         self.weights = Nk / n_samples
        
#         # Update means
#         self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
        
#         # Update covariances
#         for k in range(self.n_components):
#             diff = X - self.means[k]
#             self.covs[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]

#     def fit(self, X):
#         self.initialize_parameters(X)
#         log_likelihood_old = None
        
#         for iteration in range(self.max_iter):
#             # E-step
#             responsibilities = self.e_step(X)
            
#             # M-step
#             self.m_step(X, responsibilities)
            
#             # Compute log likelihood
#             log_likelihood = self.compute_log_likelihood(X)
#             if log_likelihood_old is not None and abs(log_likelihood - log_likelihood_old) < self.tol:
#                 break
#             log_likelihood_old = log_likelihood
        
#         return self

#     def compute_log_likelihood(self, X):
#         n_samples = X.shape[0]
#         likelihood = np.zeros(n_samples)
#         for k in range(self.n_components):
#             gaussian = multivariate_normal(mean=self.means[k], cov=self.covs[k])
#             likelihood += self.weights[k] * gaussian.pdf(X)
#         return np.sum(np.log(likelihood))

#     def predict(self, X):
#         responsibilities = self.e_step(X)
#         return np.argmax(responsibilities, axis=1)

# # Example usage (for testing):
# if __name__ == "__main__":
#     # Generate synthetic data
#     np.random.seed(42)
#     X = np.vstack([np.random.normal(0, 1, (100, 2)), np.random.normal(5, 1, (100, 2))])
#     gmm = CustomGMM(n_components=2, random_state=42)
#     gmm.fit(X)
#     labels = gmm.predict(X)
#     print("Cluster labels:", labels[:10])
















# # Import necessary libraries
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # from custom_gmm import CustomGMM  # Import the custom GMM implementation

# from custom_gmm import EnhancedGMM  # Import the custom GMM implementation

# ### Step 1: Data Collection
# # Fetch S&P 500 historical data from Yahoo Finance
# print("Fetching S&P 500 data...")
# stock = yf.download('^GSPC', start='2019-01-01', end='2024-01-01')
# stock = stock[['Close']]  # Retain only closing prices
# print(f"Data fetched: {stock.shape[0]} days of closing prices.")

# ### Step 2: Feature Engineering
# # Calculate daily returns
# stock['Return'] = stock['Close'].pct_change().dropna()

# # Calculate 5-day rolling volatility
# stock['Volatility'] = stock['Return'].rolling(window=5).std()

# # Remove rows with NaN values
# stock = stock.dropna()
# print(f"Features engineered: {stock.shape[0]} days with returns and volatility.")

# # Prepare feature array for clustering
# features = stock[['Return', 'Volatility']].values

# ### Step 3: Model Implementation
# # Initialize and fit custom GMM with 3 clusters
# print("Fitting Custom Gaussian Mixture Model...")
# gmm = EnhancedGMM(n_components=3, random_state=42)
# gmm.fit(features)

# # Predict cluster labels for each day
# labels = gmm.predict(features)

# # Add cluster labels to the dataframe
# stock['Cluster'] = labels
# print("Clustering completed.")

# ### Step 4: Trend Definition
# # Compute mean return for each cluster
# cluster_means = stock.groupby('Cluster')['Return'].mean()
# print("\nCluster Mean Returns:")
# print(cluster_means)

# # Define trend labels based on mean returns
# trend_labels = {
#     cluster_means.idxmax(): 'Uptrend',
#     cluster_means.idxmin(): 'Downtrend',
#     cluster_means.index[~cluster_means.index.isin([cluster_means.idxmax(), cluster_means.idxmin()])][0]: 'Neutral'
# }

# # Map clusters to trend labels
# stock['Trend'] = stock['Cluster'].map(trend_labels)
# print("\nTrend assignments:")
# print(stock['Trend'].value_counts())

# ### Step 5: Evaluation
# # Visualize closing prices with trend clusters
# plt.figure(figsize=(14, 7))
# for trend in trend_labels.values():
#     subset = stock[stock['Trend'] == trend]
#     plt.scatter(subset.index, subset['Close'], label=trend, s=5)
# plt.title('S&P 500 Closing Prices with Trend Clusters (2019-2024)')
# plt.xlabel('Date')
# plt.ylabel('Closing Price')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Calculate alignment with actual price movement
# stock['Price Direction'] = np.sign(stock['Return'])
# stock['Trend Direction'] = stock['Trend'].map({'Uptrend': 1, 'Downtrend': -1, 'Neutral': 0})
# alignment = (stock['Price Direction'] == stock['Trend Direction']).mean()
# print(f"\nAlignment with actual price direction: {alignment:.2%}")