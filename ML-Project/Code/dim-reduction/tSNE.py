from sklearn.manifold import TSNE
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
    parser.add_argument("--perplexity", "-p", type=float, default=30.0, help="Perplexity parameter for t-SNE")
    parser.add_argument("--learning_rate", "-l", type=float, default=200.0, help="Learning rate for t-SNE")
    parser.add_argument("--scale", choices=["standard", "minmax", "none"], default="standard", help="Feature scaling method before t-SNE")
    args = parser.parse_args()

    train_path = "./data/train.csv"
    train_inputs, train_labels = load_dataset(train_path, labeled=True)
    train_inputs = train_inputs.astype(np.float32, copy=False)

    test_path = "./data/test.csv"
    test_inputs,_ = load_dataset(test_path, labeled=False)
    test_inputs = test_inputs.astype(np.float32, copy=False) 

    scaler = None
    if args.scale == "standard":
        scaler = StandardScaler()
    elif args.scale == "minmax":
        scaler = MinMaxScaler()

    train_size = train_inputs.shape[0]
    concat_inputs = np.concat((train_inputs, test_inputs), axis = 0)

    if scaler is not None:
        concat_inputs = scaler.fit_transform(concat_inputs).astype(np.float32, copy=False)

    tsne = TSNE(n_components=args.components, perplexity=args.perplexity,
                learning_rate=args.learning_rate, n_jobs=-1, random_state=42)
    
    new_concat_inputs = tsne.fit_transform(concat_inputs)

    new_train_inputs = new_concat_inputs[:train_size]
    new_test_inputs = new_concat_inputs[train_size:]
    save_dataset(f"./data/reduced/tSNE/train-{args.components}-{args.perplexity}.csv", new_train_inputs, train_labels)
    save_dataset(f"./data/reduced/tSNE/test-{args.components}-{args.perplexity}.csv", new_test_inputs)