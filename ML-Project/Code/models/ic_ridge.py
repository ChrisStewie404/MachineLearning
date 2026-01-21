import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import sys
import os
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
import util

def main():
    # Define data path
    data_path = REPO_ROOT / 'data' / 'train.csv'
    
    # Check if file exists
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        # Try sampled
        data_path = REPO_ROOT / 'data' / 'train-sampled.csv'
        if not data_path.exists():
             print(f"Error: {data_path} not found either.")
             return
        print(f"Using {data_path} instead.")

    print(f"Loading data from {data_path}...")
    try:
        X, y = util.load_dataset(str(data_path), labeled=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Data loaded. Shape: X={X.shape}, y={y.shape}")

    # Scale data
    print("Scaling data...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    n_samples, n_features = X.shape

    # Compute singular values for effective degrees of freedom calculation
    # X = U S V^T
    # eigenvalues of X^T X are S^2
    print("Computing SVD of X...")
    S = np.linalg.svd(X, compute_uv=False)
    S2 = S ** 2

    # Define alphas (regularization strength)
    # Ridge alphas are typically larger than Lasso alphas.
    # We'll use a log scale to cover a wide range.
    alphas = np.logspace(-2, 5, 100)
    
    aic_scores = []
    bic_scores = []
    
    print("Computing Ridge path...")
    for alpha in alphas:
        # Fit Ridge
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        
        # Predict and calculate RSS
        y_pred = model.predict(X)
        rss = np.sum((y - y_pred) ** 2)
        
        # Calculate effective degrees of freedom (df)
        # df = sum( d_i^2 / (d_i^2 + alpha) )
        df = np.sum(S2 / (S2 + alpha))
        
        # Estimate noise variance (sigma^2)
        # Usually estimated from a low-bias model (e.g. very small alpha or OLS)
        # Here we can use the RSS of the current model / (n - df) or a fixed estimate.
        # LassoLarsIC uses an estimate from the full model (OLS) if n > p, or similar.
        # Let's use the OLS estimate (alpha -> 0) if possible, or just the current model's RSS/n approximation for the likelihood term.
        # Standard AIC formula: n * log(RSS/n) + 2 * df
        # Standard BIC formula: n * log(RSS/n) + df * log(n)
        # This assumes constant variance and drops constant terms.
        
        aic = n_samples * np.log(rss / n_samples) + 2 * df
        bic = n_samples * np.log(rss / n_samples) + df * np.log(n_samples)
        
        aic_scores.append(aic)
        bic_scores.append(bic)

    aic_scores = np.array(aic_scores)
    bic_scores = np.array(bic_scores)
    
    # Find best alpha
    best_alpha_aic = alphas[np.argmin(aic_scores)]
    best_alpha_bic = alphas[np.argmin(bic_scores)]
    
    print(f"Best Alpha AIC: {best_alpha_aic}")
    print(f"Best Alpha BIC: {best_alpha_bic}")

    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(alphas, aic_scores, label='AIC')
    plt.plot(alphas, bic_scores, label='BIC')
    
    # Add vertical lines for best alpha
    plt.axvline(best_alpha_aic, linestyle='--', color='tab:blue', label=f'AIC Best Alpha: {best_alpha_aic:.4f}')
    plt.axvline(best_alpha_bic, linestyle='--', color='tab:orange', label=f'BIC Best Alpha: {best_alpha_bic:.4f}')

    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Criterion Score')
    plt.title('AIC and BIC vs Alpha for Ridge')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    # Save figure
    output_dir = REPO_ROOT / 'res' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'ridge_ic.png'
    
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    main()
