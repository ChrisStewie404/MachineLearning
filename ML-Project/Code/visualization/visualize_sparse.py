import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# add repo root so util.py can be imported even when cwd differs
REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust parents if nested deeper
sys.path.append(str(REPO_ROOT))
from util import load_dataset
import numpy as np

if __name__ == "__main__":
    data, labels = load_dataset("../data/train.csv", labeled=True)    

    plt.figure(figsize=(8, 6))
    plt.hist(np.abs(data.flatten()), bins=50, color='skyblue', log=True)
    # plt.title("Histogram of Feature Magnitudes", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Absolute Feature Value", fontsize=12, fontweight='medium')
    plt.ylabel("Log Frequency", fontsize=12, fontweight='medium')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()