import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import argparse

# add repo root so util.py can be imported even when cwd differs
REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust parents if nested deeper
sys.path.append(str(REPO_ROOT))
import util
import numpy as np
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, required=True)
    args = parser.parse_args()

    rf = util.load_model(args.model_path)

    plt.figure(figsize=(18, 8))
    plot_tree(
        rf.estimators_[0],
        max_depth=2,
        filled=True,
        impurity=False,
        proportion=False,
        rounded=True,
        fontsize=8,
        label="none"
    )
    plt.title("Visualization of the First Tree in Random Forest", fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()