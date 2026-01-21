from sklearn.decomposition._pca import PCA
import argparse
import os
import sys
from pathlib import Path

# add repo root so util.py can be imported even when cwd differs
REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust parents if nested deeper
sys.path.append(str(REPO_ROOT))
from util import load_dataset, save_dataset
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", "-c", type=int)
    args = parser.parse_args()

    train_path = "../data/train.csv"
    train_inputs, train_labels = load_dataset(train_path, labeled=True)
    pca = PCA(n_components=args.components)
    pca.fit(train_inputs)
    
    new_train_inputs = pca.transform(train_inputs)
    print(new_train_inputs.shape)
    save_dataset(f"../data/reduced/PCA/train-{args.components}.csv", new_train_inputs, train_labels)

    test_path = "../data/test.csv"
    test_inputs,_ = load_dataset(test_path, labeled=False)

    new_test_inputs = pca.transform(test_inputs)
    print(new_test_inputs.shape)
    save_dataset(f"../data/reduced/PCA/test-{args.components}.csv", new_test_inputs)