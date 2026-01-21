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
    train_path = "../data/train.csv"
    train_inputs, train_labels = load_dataset(train_path, labeled=True)

    class_counts = np.bincount(train_labels.astype(int))
    present_classes = np.flatnonzero(class_counts)
    most_class = int(np.argmax(class_counts))
    least_class = int(present_classes[np.argmin(class_counts[present_classes])])
    print(f"Most frequent class: {most_class} ({int(class_counts[most_class])} samples)")
    print(f"Least frequent class: {least_class} ({int(class_counts[least_class])} samples)")

    # max_label = np.max(train_labels)
    
    # plt.figure()
    # plt.hist(train_labels, bins = max_label+1, color="gold")
    # plt.xlabel("class")
    # plt.ylabel("density")
    # # plt.title("Class Distribution of Train Data", fontweight="bold")
    # plt.show()
    