# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

### Step 1: Data Collection
# Fetch S&P 500 historical data from Yahoo Finance
print("Fetching S&P 500 data...")
stock = yf.download('^GSPC', start='2019-01-01', end='2024-01-01')
stock = stock[['Close']]  # Retain only closing prices
print(f"Data fetched: {stock.shape[0]} days of closing prices.")

### Step 0: Plot Raw Data (Newly added)
plt.figure(figsize=(14, 7))
plt.plot(stock.index, stock['Close'])  # Plotting closing prices
plt.title('S&P 500 Closing Prices (2019-2024)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.grid(True)
plt.show()

### Step 2: Feature Engineering
# Calculate daily returns
stock['Return'] = stock['Close'].pct_change().dropna()

# Calculate 5-day rolling volatility
stock['Volatility'] = stock['Return'].rolling(window=5).std()

# Remove rows with NaN values
stock = stock.dropna()
print(f"Features engineered: {stock.shape[0]} days with returns and volatility.")

# Prepare feature array for clustering
features = stock[['Return', 'Volatility']].values

### Step 3: Model Implementation
# Initialize and fit GMM with 3 clusters
print("Fitting Gaussian Mixture Model...")
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(features)

# Predict cluster labels for each day
labels = gmm.predict(features)

# Add cluster labels to the dataframe
stock['Cluster'] = labels
print("Clustering completed.")

### Step 4: Trend Definition
# Compute mean return for each cluster
cluster_means = stock.groupby('Cluster')['Return'].mean()
print("\nCluster Mean Returns:")
print(cluster_means)

# Define trend labels based on mean returns
trend_labels = {
    cluster_means.idxmax(): 'Uptrend',
    cluster_means.idxmin(): 'Downtrend',
    cluster_means.index[~cluster_means.index.isin([cluster_means.idxmax(), cluster_means.idxmin()])][0]: 'Neutral'
}

# Map clusters to trend labels
stock['Trend'] = stock['Cluster'].map(trend_labels)
print("\nTrend assignments:")
print(stock['Trend'].value_counts())

### Step 5: Evaluation
# Visualize closing prices with trend clusters
plt.figure(figsize=(14, 7))
for trend in trend_labels.values():
    subset = stock[stock['Trend'] == trend]
    plt.scatter(subset.index, subset['Close'], label=trend, s=5)
plt.title('S&P 500 Closing Prices with Trend Clusters (2019-2024)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()

# Calculate alignment with actual price movement
stock['Price Direction'] = np.sign(stock['Return'])
stock['Trend Direction'] = stock['Trend'].map({'Uptrend': 1, 'Downtrend': -1, 'Neutral': 0})
alignment = (stock['Price Direction'] == stock['Trend Direction']).mean()
print(f"\nAlignment with actual price direction: {alignment:.2%}")

### Notes for Deliverables
print("\nNext Steps for Deliverables:")
print("- Save this script as your code deliverable.")
print("- Use the plot and alignment metric in your report.")
print("- Write a 2-3 page report covering: Introduction, Data, Model, Results, and Conclusion.")