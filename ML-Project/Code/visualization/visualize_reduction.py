"""
Visualize high-dimensional data after dimension reduction.
Supports 2D and 3D visualizations with optional class labels.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# add repo root so util.py can be imported even when cwd differs
REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust parents if nested deeper
sys.path.append(str(REPO_ROOT))
from util import load_dataset
from matplotlib.lines import Line2D


def visualize_2d(data, labels=None, title=None, save_path=None, class_counts=None):
    """Create a 2D scatter plot of the reduced data.
    
    Args:
        data: Nx2 numpy array of reduced features
        labels: Optional Nx1 numpy array of class labels
        title: Plot title
        save_path: Optional path to save the figure
        class_counts: Optional dict mapping class labels to their counts
    """
    plt.figure(figsize=(12, 8))
    
    if labels is not None:
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # Use better colormap for more classes
        if n_classes <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_classes]
        elif n_classes <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_classes]
        else:
            colors = plt.cm.gist_ncar(np.linspace(0, 1, n_classes))
        
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            count_str = f" (n={class_counts[label]})" if class_counts and label in class_counts else ""
            plt.scatter(data[mask, 0], data[mask, 1], 
                       c=[colors[idx]], label=f'Class {int(label)}{count_str}',
                       alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
        
        ncol = 2 if n_classes > 20 else 1
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8, 
                  framealpha=0.95, ncol=ncol, title='Classes', title_fontsize=9)
    else:
        plt.scatter(data[:, 0], data[:, 1], alpha=0.7, s=30, 
                   c='steelblue', edgecolors='white', linewidth=0.5)
    
    plt.xlabel('Component 1', fontsize=13, fontweight='medium')
    plt.ylabel('Component 2', fontsize=13, fontweight='medium')
    ax = plt.gca()
    max_range = (data.max(axis=0) - data.min(axis=0)).max()
    if max_range > 0:
        mid_x = 0.5 * (data[:, 0].max() + data[:, 0].min())
        mid_y = 0.5 * (data[:, 1].max() + data[:, 1].min())
        half_range = max_range * 0.5
        ax.set_xlim(mid_x - half_range, mid_x + half_range)
        ax.set_ylim(mid_y - half_range, mid_y + half_range)
    ax.set_aspect("equal", adjustable="box")
    if title:
        plt.title(title, fontsize=15, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.25, linestyle='--')
    plt.subplots_adjust(right=0.82)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D plot to {save_path}")
    
    plt.show()

def visualize_2ds(data_arr, labels=None, title=None, save_path=None, class_counts=None):

    datasets = list(data_arr) if isinstance(data_arr, (list, tuple)) else [data_arr]
    arrays = [np.asarray(d) for d in datasets]
    if any(a.ndim != 2 or a.shape[1] < 2 for a in arrays):
        raise ValueError("Each dataset must be a 2D array with at least two columns.")

    n_panels = len(arrays)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes.ravel()

    labels = np.asarray(labels).ravel() if labels is not None else None
    legend_handles = None
    legend_labels = None

    if labels is not None:
        if labels.shape[0] != arrays[0].shape[0]:
            raise ValueError("Label count must match the number of samples.")
        unique_labels = np.unique(labels)
        n_classes = unique_labels.size
        if n_classes <= 10:
            cmap = plt.cm.get_cmap("tab10", n_classes)
        elif n_classes <= 20:
            cmap = plt.cm.get_cmap("tab20", n_classes)
        else:
            cmap = plt.cm.get_cmap("gist_ncar", n_classes)
        palette = cmap(np.arange(n_classes))
        legend_handles = []
        legend_labels = []
        for idx, cls in enumerate(unique_labels):
            display_name = f"Class {int(cls)}"
            if class_counts and cls in class_counts:
                display_name += f" (n={class_counts[cls]})"
            legend_handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=6,
                                            markerfacecolor=palette[idx], markeredgecolor="white",
                                            markeredgewidth=0.5, alpha=0.7))
            legend_labels.append(display_name)
    else:
        palette = None

    stacked = np.vstack(arrays)
    ranges = stacked.max(axis=0) - stacked.min(axis=0)
    max_span = ranges.max()
    if max_span > 0:
        center = 0.5 * (stacked.max(axis=0) + stacked.min(axis=0))
        limits = np.column_stack((center - 0.5 * max_span, center + 0.5 * max_span))
    else:
        limits = np.column_stack((stacked[:, 0], stacked[:, 0]))

    per_plot_titles = None
    global_title = None
    if isinstance(title, (list, tuple)):
        per_plot_titles = [str(t) for t in title]
    elif title is not None:
        global_title = title

    for idx, (ax, arr) in enumerate(zip(axes, arrays)):
        if labels is None:
            ax.scatter(arr[:, 0], arr[:, 1], c="steelblue", alpha=0.7, s=30,
                        edgecolors="white", linewidths=0.5)
        else:
            for color_idx, cls in enumerate(np.unique(labels)):
                mask = labels == cls
                ax.scatter(arr[mask, 0], arr[mask, 1], alpha=0.7, s=30,
                            edgecolors="white", linewidths=0.5,
                            c=[palette[color_idx]])
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_xlabel("Component 1", fontsize=12, fontweight="medium")
        if idx == 0:
            ax.set_ylabel("Component 2", fontsize=12, fontweight="medium")
        if per_plot_titles and idx < len(per_plot_titles):
            ax.set_title(per_plot_titles[idx], fontsize=13, fontweight="bold")

    if global_title:
        fig.suptitle(global_title, fontsize=15, fontweight="bold", y=0.98)

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), framealpha=0.95, fontsize=9, title="Classes",
                    title_fontsize=10)

    fig.tight_layout(rect=[0, 0, 0.96, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved 2D comparison plot to {save_path}")

    plt.show()

def visualize_3d(data, labels=None, title=None, save_path=None, class_counts=None):
    """Create a 3D scatter plot of the reduced data.
    
    Args:
        data: Nx3 numpy array of reduced features
        labels: Optional Nx1 numpy array of class labels
        title: Plot title
        save_path: Optional path to save the figure
        class_counts: Optional dict mapping class labels to their counts
    """
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # Use better colormap for more classes
        if n_classes <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_classes]
        elif n_classes <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_classes]
        else:
            colors = plt.cm.gist_ncar(np.linspace(0, 1, n_classes))
        
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            count_str = f" (n={class_counts[label]})" if class_counts and label in class_counts else ""
            ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2],
                      c=[colors[idx]], label=f'Class {int(label)}{count_str}',
                      alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
        
        ncol = 2 if n_classes > 10 else 1
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8, 
                 framealpha=0.95, ncol=ncol, title='Classes', title_fontsize=9)
    else:
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
                  alpha=0.7, s=30, c='steelblue', edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Component 1', fontsize=12, fontweight='medium')
    ax.set_ylabel('Component 2', fontsize=12, fontweight='medium')
    ax.set_zlabel('Component 3', fontsize=12, fontweight='medium')
    if title:
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.25, linestyle='--')
    plt.subplots_adjust(right=0.82)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D plot to {save_path}")
    
    plt.show()

def visualize_3ds(data_arr, labels=None, title=None, save_path=None, class_counts=None):
    datasets = list(data_arr) if isinstance(data_arr, (list, tuple)) else [data_arr]
    arrays = [np.asarray(d) for d in datasets]
    if any(a.ndim != 2 or a.shape[1] < 3 for a in arrays):
        raise ValueError("Each dataset must be a 2D array with at least three columns.")

    n_panels = len(arrays)
    fig = plt.figure(figsize=(7 * n_panels, 6))
    axes = [fig.add_subplot(1, n_panels, i + 1, projection='3d') for i in range(n_panels)]

    labels = np.asarray(labels).ravel() if labels is not None else None
    legend_handles = None
    legend_labels = None

    if labels is not None:
        if labels.shape[0] != arrays[0].shape[0]:
            raise ValueError("Label count must match the number of samples.")
        unique_labels = np.unique(labels)
        n_classes = unique_labels.size
        if n_classes <= 10:
            cmap = plt.cm.get_cmap("tab10", n_classes)
        elif n_classes <= 20:
            cmap = plt.cm.get_cmap("tab20", n_classes)
        else:
            cmap = plt.cm.get_cmap("gist_ncar", n_classes)
        palette = cmap(np.arange(n_classes))
        legend_handles = []
        legend_labels = []
        for idx, cls in enumerate(unique_labels):
            display_name = f"Class {int(cls)}"
            if class_counts and cls in class_counts:
                display_name += f" (n={class_counts[cls]})"
            legend_handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=6,
                                            markerfacecolor=palette[idx], markeredgecolor="white",
                                            markeredgewidth=0.5, alpha=0.7))
            legend_labels.append(display_name)
    else:
        palette = None

    stacked = np.vstack(arrays)
    ranges = stacked.max(axis=0) - stacked.min(axis=0)
    max_span = ranges.max()
    if max_span > 0:
        center = 0.5 * (stacked.max(axis=0) + stacked.min(axis=0))
        limits = np.column_stack((center - 0.5 * max_span, center + 0.5 * max_span))
    else:
        limits = np.column_stack((stacked[:, 0], stacked[:, 0]))
    per_plot_titles = None
    global_title = None
    if isinstance(title, (list, tuple)):
        per_plot_titles = [str(t) for t in title]
    elif title is not None:
        global_title = title
    for idx, (ax, arr) in enumerate(zip(axes, arrays)):
        if labels is None:
            ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c="steelblue", alpha=0.7, s=30,
                        edgecolors="white", linewidths=0.5)
        else:
            for color_idx, cls in enumerate(np.unique(labels)):
                mask = labels == cls
                ax.scatter(arr[mask, 0], arr[mask, 1], arr[mask, 2], alpha=0.7, s=30,
                            edgecolors="white", linewidths=0.5,
                            c=[palette[color_idx]])
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_zlim(limits[2])
        ax.set_xlabel("Component 1", fontsize=12, fontweight="medium")
        ax.set_ylabel("Component 2", fontsize=12, fontweight="medium")
        ax.set_zlabel("Component 3", fontsize=12, fontweight="medium")
        if per_plot_titles and idx < len(per_plot_titles):
            ax.set_title(per_plot_titles[idx], fontsize=13, fontweight="bold")
    if global_title:
        fig.suptitle(global_title, fontsize=15, fontweight="bold", y=0.98)
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), framealpha=0.95, fontsize=9, title="Classes",
                    title_fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.96, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved 3D comparison plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize dimension-reduced data")
    parser.add_argument("--input", "-i", type=str, required=True, 
                       help="Path to the reduced data CSV file")
    parser.add_argument("--labeled", action="store_true", 
                       help="Whether the data includes labels (last column)")
    parser.add_argument("--title", "-t", type=str, default="none",
                       help="Plot title")
    parser.add_argument("--save", "-s", type=str, default=None,
                       help="Path to save the plot (e.g., ./plots/isomap_2d.png)")
    parser.add_argument("--max_samples", "-n", type=int, default=None,
                       help="Maximum number of samples to plot (for performance)")
    parser.add_argument("--top_n", "-cl", type=int, default=None,
                       help="Number of most frequent classes to visualize (filters by class frequency)")
    
    args = parser.parse_args()
    
    # Load the data
    print(f"Loading data from {args.input}...")
    data, labels = load_dataset(args.input, labeled=args.labeled)
    
    class_counts = None
    
    # Filter by top N most frequent classes if requested
    if args.labeled and args.top_n is not None:
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\nTotal classes in dataset: {len(unique_labels)}")
        
        # Get top N most frequent classes
        top_indices = np.argsort(counts)[-args.top_n:][::-1]
        top_classes = unique_labels[top_indices]
        top_counts = counts[top_indices]
        
        # Create class count dictionary
        class_counts = dict(zip(top_classes, top_counts))
        
        # Filter data
        mask = np.isin(labels, top_classes)
        original_size = len(data)
        data = data[mask]
        labels = labels[mask]
        
        print(f"Filtered to top {args.top_n} most frequent classes:")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  Class {int(cls)}: {cnt} samples ({100*cnt/original_size:.1f}%)")
        print(f"Total samples after filtering: {len(data)} / {original_size} ({100*len(data)/original_size:.1f}%)\n")
    
    # Subsample if requested
    if args.max_samples and data.shape[0] > args.max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(data.shape[0], args.max_samples, replace=False)
        data = data[indices]
        if args.labeled:
            labels = labels[indices]
        print(f"Subsampled {args.max_samples} points for visualization")
    
    print(f"Data shape: {data.shape}")
    
    # Determine visualization type based on number of components
    n_components = data.shape[1]
    labels_to_plot = labels if args.labeled else None
    args.title = args.title if args.title != "none" else None

    if n_components == 2:
        print("Creating 2D visualization...")
        visualize_2d(data, labels_to_plot, args.title, args.save, class_counts)
    elif n_components == 3:
        print("Creating 3D visualization...")
        visualize_3d(data, labels_to_plot, args.title, args.save, class_counts)
    elif n_components > 3:
        print(f"Data has {n_components} components. Visualizing first 3 dimensions...")
        visualize_3d(data[:, :3], labels_to_plot, 
                    f"{args.title} (First 3 Components)", args.save, class_counts)
    else:
        print(f"Error: Cannot visualize data with {n_components} component(s)")
        return
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
