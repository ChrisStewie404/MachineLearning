from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse
import numpy as np
import os
import sys
from pathlib import Path

# add repo root so util.py can be imported even when cwd differs
REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust parents if nested deeper
sys.path.append(str(REPO_ROOT))
from util import load_dataset, save_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", "-c", type=int, required=True, help="Target embedding dimension")
    parser.add_argument("--neighbors", "-k", type=int, default=10, help="Number of neighbors for the geodesic graph")
    parser.add_argument("--max_samples", type=int, default=5000, help="Subsample size for fitting ISOMAP to avoid O(N^2) blow-up")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument("--scale", choices=["standard", "minmax", "none"], default="standard", help="Feature scaling method before ISOMAP")
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_state)

    train_path = "./data/train.csv"
    train_inputs, train_labels = load_dataset(train_path, labeled=True)
    train_inputs = train_inputs.astype(np.float32, copy=False)

    # Scale features to improve neighbor graph quality and numerical stability.
    scaler = None
    if args.scale == "standard":
        scaler = StandardScaler()
    elif args.scale == "minmax":
        scaler = MinMaxScaler()

    if scaler is not None:
        train_inputs = scaler.fit_transform(train_inputs).astype(np.float32, copy=False)

    # Fit on a manageable subset to avoid dense NxN distance matrices exhausting memory.
    fit_inputs = train_inputs
    if args.max_samples and fit_inputs.shape[0] > args.max_samples:
        subset_idx = rng.choice(fit_inputs.shape[0], args.max_samples, replace=False)
        fit_inputs = fit_inputs[subset_idx]
        print(f"Subsampled {fit_inputs.shape[0]} points (from {train_inputs.shape[0]}) for ISOMAP fitting")

    isomap = Isomap(n_components=args.components, n_neighbors=args.neighbors, n_jobs=-1)
    isomap.fit(fit_inputs)

    new_train_inputs = isomap.transform(train_inputs)
    print(new_train_inputs.shape)
    save_dataset(f"./data/reduced/ISOMAP/train-{args.max_samples}-{args.components}-{args.neighbors}.csv", new_train_inputs, train_labels)

    test_path = "./data/test.csv"
    test_inputs,_ = load_dataset(test_path, labeled=False)
    test_inputs = test_inputs.astype(np.float32, copy=False)

    if scaler is not None:
        test_inputs = scaler.transform(test_inputs).astype(np.float32, copy=False)

    new_test_inputs = isomap.transform(test_inputs)
    print(new_test_inputs.shape)
    save_dataset(f"./data/reduced/ISOMAP/test-{args.max_samples}-{args.components}-{args.neighbors}.csv", new_test_inputs)
