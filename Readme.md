# Enhanced Gaussian Mixture Model (GMM) Project

This project implements an Enhanced Gaussian Mixture Model (GMM) for clustering and trend analysis of financial data. The project uses historical S&P 500 data to demonstrate the model's capabilities.

---

## Features
- Fetches historical stock data using `yfinance`.
- Performs feature engineering (daily returns and rolling volatility).
- Implements an enhanced GMM with visualization and validation features.
- Provides clustering and trend analysis for financial data.
- Includes visualization of clustering results and log-likelihood convergence.

---

## Requirements
To set up and run this project, you need the following libraries:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `yfinance`
- `tabulate`
- `openpyxl`

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment
- On **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies
Install all required libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 5. Run the Project
- To run the main script:
  ```bash
  python main-custom.py
  ```
- To test the `EnhancedGMM` implementation:
  ```bash
  python custom_gmm.py
  ```

---

## File Structure
- `main-custom.py`: Main script for fetching data, feature engineering, and clustering.
- `custom_gmm.py`: Implementation of the Enhanced Gaussian Mixture Model.
- `requirements.txt`: List of required Python libraries.

---

## Visualizations
The project generates the following visualizations:
1. Raw S&P 500 closing prices.
2. Clustering results in 2D feature space.
3. Log-likelihood convergence plot.

---

## Example Usage
The `custom_gmm.py` file includes an example of using the Enhanced GMM with synthetic data. You can modify the `main-custom.py` script to analyze other datasets.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Author
- **Your Name**  
  [Your GitHub Profile](https://github.com/harsh-5401)

---

## Acknowledgments
- Yahoo Finance for providing historical stock data.
- Scikit-learn for inspiration on Gaussian Mixture Models.