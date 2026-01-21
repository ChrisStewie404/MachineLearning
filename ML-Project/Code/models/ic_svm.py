import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelBinarizer
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
    
    # Convert labels to -1, 1 for SVM (Multi-class handling)
    lb = LabelBinarizer()
    Y_bin = lb.fit_transform(y)
    # If binary, LabelBinarizer returns (n, 1), but LinearSVC decision_function returns (n,)
    # We need to handle shapes carefully.
    # For this specific dataset, we know it's likely multiclass (100 classes).
    # Y_bin will be (n, 100) with 0/1.
    y_svm = 2 * Y_bin - 1

    # Define alphas (regularization strength). 
    # In LinearSVC, C is inverse of regularization strength.
    # We will treat alpha ~ 1/C.
    # We use a range similar to the Lasso plot [0, 0.1] but we need to be careful with C.
    # If alpha is very small, C is very large.
    # Let's pick a range of alphas.
    alphas = np.linspace(0.001, 0.1, 50)
    
    aic_scores = []
    bic_scores = []
    real_alphas = []

    n_samples = X.shape[0]

    print("Computing SVM path...")
    for alpha in alphas:
        # C = 1 / (n_samples * alpha) to be somewhat consistent with Lasso formulation
        # or just C = 1/alpha. Let's use C = 1 / (alpha * n_samples) to match sklearn Lasso scale roughly?
        # Sklearn Lasso: 1/(2n) * RSS + alpha * L1.
        # LinearSVC L1: L1 + C * Loss.
        # Equivalent: 1/C * L1 + Loss.
        # So 1/C ~ alpha.
        # If we want to match the scale, we might need to adjust, but let's just use C = 1/alpha for simplicity
        # and note that the scale might differ.
        # Actually, let's use C = 1 / (alpha * n_samples) to keep C in a reasonable range for convergence
        # if alpha is small.
        # If alpha = 0.01, n=20000, C = 1/200 = 0.005.
        # If alpha = 0.001, C = 0.05.
        # If alpha = 0.1, C = 0.0005.
        # This seems too small for C.
        # Let's just use C = 1/alpha for now, assuming alpha is "regularization parameter".
        # If alpha = 0.01, C = 100.
        
        if alpha <= 0:
            continue
            
        C = 1.0 / (alpha * n_samples) # Scaling by n_samples often helps with convergence stability
        # But let's try a simpler C = 1/alpha first? 
        # No, let's stick to a range that works.
        # Let's try C = 1 / (alpha * 100) or just experiment.
        # Let's use C = 1 / (alpha * n_samples) as a starting point for "Lasso-like" scaling.
        
        # Actually, let's just use a logspace for C and then calculate alpha = 1/C (normalized) for plotting.
        # But user wants X-axis to be alpha.
        
        # Let's use C = 1 / (alpha * n_samples)
        C = 1.0 / (alpha * n_samples)

        # LinearSVC with L1 penalty requires squared_hinge loss and dual=False
        clf = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=C, max_iter=2000, tol=1e-4)
        try:
            clf.fit(X, y)
        except Exception as e:
            print(f"Fit failed for alpha={alpha}: {e}")
            continue
            
        k = np.sum(clf.coef_ != 0)
        
        # Calculate RSS of decision function
        # decision_function outputs distance to hyperplane.
        # We want to see how well it fits the labels {-1, 1}.
        df = clf.decision_function(X)
        
        if df.ndim == 1:
             # Binary case
             # y_svm should be (n_samples, 1) or (n_samples,)
             # Ensure shapes match
             if y_svm.ndim == 2 and y_svm.shape[1] == 1:
                 y_svm_flat = y_svm.ravel()
             else:
                 y_svm_flat = y_svm
             
             hinge_losses = np.maximum(0, 1 - y_svm_flat * df)
             rss = np.sum(hinge_losses ** 2)
        else:
             # Multiclass case
             # y_svm is (n_samples, n_classes)
             # df is (n_samples, n_classes)
             hinge_losses = np.maximum(0, 1 - y_svm * df)
             rss = np.sum(hinge_losses ** 2)
        
        # AIC / BIC
        # AIC = n * log(RSS/n) + 2k
        # BIC = n * log(RSS/n) + k * log(n)
        
        aic = n_samples * np.log(rss / n_samples) + 2 * k
        bic = n_samples * np.log(rss / n_samples) + k * np.log(n_samples)
        
        aic_scores.append(aic)
        bic_scores.append(bic)
        real_alphas.append(alpha)

    aic_scores = np.array(aic_scores)
    bic_scores = np.array(bic_scores)
    real_alphas = np.array(real_alphas)

    # Find best alpha
    best_alpha_aic = real_alphas[np.argmin(aic_scores)]
    best_alpha_bic = real_alphas[np.argmin(bic_scores)]

    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(real_alphas, aic_scores, label='AIC')
    plt.plot(real_alphas, bic_scores, label='BIC')
    
    # Add vertical lines for best alpha
    plt.axvline(best_alpha_aic, linestyle='--', color='tab:blue', label=f'AIC Best Alpha: {best_alpha_aic:.4f}')
    plt.axvline(best_alpha_bic, linestyle='--', color='tab:orange', label=f'BIC Best Alpha: {best_alpha_bic:.4f}')

    # Mark best alphas on X-axis
    ax = plt.gca()
    # ax.set_xlim(0, 0.1)
    current_ticks = list(ax.get_xticks())
    new_ticks = sorted(list(set(current_ticks + [best_alpha_aic, best_alpha_bic])))
    # Filter ticks to be within range
    min_a, max_a = min(real_alphas), max(real_alphas)
    new_ticks = [t for t in new_ticks if min_a <= t <= max_a]
    
    plt.xticks(new_ticks, rotation=45)

    plt.xlabel('Alpha (approx 1/(n*C))')
    plt.ylabel('Criterion Score (Proxy)')
    plt.title('AIC and BIC vs Alpha for Linear SVM (L1)')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    output_dir = REPO_ROOT / 'res' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'svm_ic.png'
    
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    main()
