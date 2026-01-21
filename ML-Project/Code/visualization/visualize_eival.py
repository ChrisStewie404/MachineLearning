import os
import sys
from pathlib import Path

# add repo root so util.py can be imported even when cwd differs
REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust parents if nested deeper
sys.path.append(str(REPO_ROOT))
from util import load_dataset
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    eigenvalues_path = "../data/reduced/PCA/eigenvalues-50.csv"
    eigenvalues = np.loadtxt(eigenvalues_path, delimiter=",")

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', color='teal')
    # plt.title("Eigenvalues from PCA", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Principal Component", fontsize=12, fontweight='medium')
    plt.ylabel("Eigenvalue", fontsize=12, fontweight='medium')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()