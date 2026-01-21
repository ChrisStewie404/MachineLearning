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

    scales = np.max(data, axis=1) - np.min(data, axis=1)

    plt.figure(figsize=(8, 6))
    plt.hist(scales, bins=50, color='salmon', log=True)
    # plt.title("Histogram of Feature Scales", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Scale (Max - Min)", fontsize=12, fontweight='medium')
    plt.ylabel("Log Frequency", fontsize=12, fontweight='medium')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()