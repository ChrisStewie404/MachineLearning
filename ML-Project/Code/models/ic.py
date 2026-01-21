import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoLarsIC
from sklearn.preprocessing import StandardScaler
import sys
import os
from pathlib import Path
import argparse

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

    # Scale data as Lasso is sensitive to scale
    print("Scaling data...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # LassoLarsIC for AIC
    print("Computing Lasso path with AIC...")
    # LassoLarsIC computes the regularization path using the LARS algorithm.
    model_aic = LassoLarsIC(criterion='aic')
    model_aic.fit(X, y)
    
    # LassoLarsIC for BIC
    print("Computing Lasso path with BIC...")
    model_bic = LassoLarsIC(criterion='bic')
    model_bic.fit(X, y)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Filter for alpha in [0, 1]
    mask_aic = (model_aic.alphas_ >= 0) & (model_aic.alphas_ <= 0.1)
    mask_bic = (model_bic.alphas_ >= 0) & (model_bic.alphas_ <= 0.1)

    # Plot AIC
    plt.plot(model_aic.alphas_[mask_aic], model_aic.criterion_[mask_aic], label='AIC')
    
    # Plot BIC
    # Note: alphas_ should be the same for both if the data is the same, 
    # but we plot against the specific alphas_ of each model to be safe.
    plt.plot(model_bic.alphas_[mask_bic], model_bic.criterion_[mask_bic], label='BIC')
    
    # Add vertical lines for best alpha
    plt.axvline(model_aic.alpha_, linestyle='--', color='tab:blue', label=f'AIC Best Alpha: {model_aic.alpha_:.4f}')
    plt.axvline(model_bic.alpha_, linestyle='--', color='tab:orange', label=f'BIC Best Alpha: {model_bic.alpha_:.4f}')

    plt.xlabel('Alpha')
    plt.ylabel('Criterion Score')
    plt.title('AIC and BIC vs Alpha for Lasso')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    output_dir = REPO_ROOT / 'res' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'lasso_ic.png'
    
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    main()
