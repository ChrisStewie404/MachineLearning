import numpy as np
from scipy.spatial.distance import cdist
import seaborn as sns
import os
import sys
from pathlib import Path

# add repo root so util.py can be imported even when cwd differs
REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust parents if nested deeper
sys.path.append(str(REPO_ROOT))
from util import load_dataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_path = "../data/train.csv"
    data, labels = load_dataset(data_path, labeled=True)
    
    centroids = np.array([data[labels == label].mean(axis=0) for label in np.unique(labels)])
    dmat = cdist(centroids, centroids, metric='euclidean')

    # replace diagonal with average intra-class distance
    for i, label in enumerate(np.unique(labels)):
        X = data[labels == label]
        if X.shape[0] <= 1:
            intra_avg = 0.0
        else:
            pairwise = cdist(X, X, metric='euclidean')
            triu = pairwise[np.triu_indices_from(pairwise, k=1)]
            intra_avg = triu.mean() if triu.size > 0 else 0.0
        dmat[i, i] = intra_avg

    plt.figure(figsize=(8,6))
    sns.heatmap(dmat, annot=False, fmt=".2f", cmap="viridis")
    plt.title("Inter/Intra Class Distance Matrix", fontsize=16)
    plt.show()