"""
Ensemble Learning - Random Forest Hyperparameter Tuning
======================================================
This module demonstrates advanced hyperparameter tuning strategies for 
RandomForestClassifier to improve accuracy on a 100-class classification task.

Key Tuning Strategies:
1. Tree-level parameters (max_depth, min_samples_split, min_samples_leaf)
2. Forest-level parameters (n_estimators, max_features, bootstrap)
3. Regularization parameters (max_samples, criterion)
4. Cross-validation for robust evaluation
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import json

# add repo root so util.py can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
import util


def baseline_random_forest(train_inputs, train_labels):
    """
    Train a baseline Random Forest to establish performance baseline.
    
    Default hyperparameters often lead to overfitting on training data.
    """
    print("\n" + "="*60)
    print("BASELINE RANDOM FOREST")
    print("="*60)
    
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    
    clf.fit(train_inputs, train_labels)
    train_accuracy = clf.score(train_inputs, train_labels)
    
    print(f"Default n_estimators=100")
    print(f"Train Accuracy: {train_accuracy*100:.2f}%")
    
    return clf, train_accuracy


def strategy_1_increase_trees(train_inputs, train_labels):
    """
    Strategy 1: Increase number of trees (n_estimators)
    
    More trees generally improve accuracy, but with diminishing returns.
    Typically 100-1000 trees are effective.
    """
    print("\n" + "="*60)
    print("STRATEGY 1: INCREASE NUMBER OF TREES (n_estimators)")
    print("="*60)
    
    n_estimators_list = [100, 200, 300, 500, 800, 1000]
    results = {}
    
    for n_est in n_estimators_list:
        clf = RandomForestClassifier(
            n_estimators=n_est,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        clf.fit(train_inputs, train_labels)
        accuracy = clf.score(train_inputs, train_labels)
        results[n_est] = accuracy
        print(f"n_estimators={n_est:4d} → Train Accuracy: {accuracy*100:.2f}%")
    
    best_n_est = max(results, key=results.get)
    print(f"\nBest n_estimators: {best_n_est} with accuracy {results[best_n_est]*100:.2f}%")
    
    return best_n_est, results


def strategy_2_tree_depth_control(train_inputs, train_labels):
    """
    Strategy 2: Control tree depth (max_depth, min_samples_split, min_samples_leaf)
    
    Deeper trees fit training data better but risk overfitting.
    Limiting depth and samples per node prevents overfitting.
    """
    print("\n" + "="*60)
    print("STRATEGY 2: CONTROL TREE DEPTH & SPLITTING")
    print("="*60)
    
    # Experiment with max_depth
    print("\n--- Testing max_depth ---")
    max_depths = [10, 15, 20, 25, 30, None]  # None = unlimited
    depth_results = {}
    
    for max_d in max_depths:
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=max_d,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(train_inputs, train_labels)
        accuracy = clf.score(train_inputs, train_labels)
        depth_results[max_d if max_d is not None else "unlimited"] = accuracy
        depth_label = max_d if max_d is not None else "unlimited"
        print(f"max_depth={depth_label:>10} → Train Accuracy: {accuracy*100:.2f}%")
    
    # Experiment with min_samples_split
    print("\n--- Testing min_samples_split ---")
    min_samples_splits = [2, 5, 10, 20, 50]
    split_results = {}
    
    for min_split in min_samples_splits:
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,  # Use a reasonable depth
            min_samples_split=min_split,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(train_inputs, train_labels)
        accuracy = clf.score(train_inputs, train_labels)
        split_results[min_split] = accuracy
        print(f"min_samples_split={min_split:2d} → Train Accuracy: {accuracy*100:.2f}%")
    
    # Experiment with min_samples_leaf
    print("\n--- Testing min_samples_leaf ---")
    min_samples_leafs = [1, 2, 4, 8, 16]
    leaf_results = {}
    
    for min_leaf in min_samples_leafs:
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=min_leaf,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(train_inputs, train_labels)
        accuracy = clf.score(train_inputs, train_labels)
        leaf_results[min_leaf] = accuracy
        print(f"min_samples_leaf={min_leaf:2d}  → Train Accuracy: {accuracy*100:.2f}%")
    
    return depth_results, split_results, leaf_results


def strategy_3_feature_sampling(train_inputs, train_labels):
    """
    Strategy 3: Optimize feature sampling (max_features)
    
    max_features controls feature diversity across trees.
    For classification: sqrt(n_features) or log2(n_features) are common.
    """
    print("\n" + "="*60)
    print("STRATEGY 3: OPTIMIZE FEATURE SAMPLING (max_features)")
    print("="*60)
    
    n_features = train_inputs.shape[1]
    max_features_options = [
        'sqrt',
        'log2',
        int(np.sqrt(n_features)),
        int(np.log2(n_features)),
        max(1, int(n_features * 0.3)),  # 30% of features
        max(1, int(n_features * 0.5)),  # 50% of features
        None  # Use all features
    ]
    
    results = {}
    for max_feat in max_features_options:
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=10,
            max_features=max_feat,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(train_inputs, train_labels)
        accuracy = clf.score(train_inputs, train_labels)
        feat_label = str(max_feat)
        results[feat_label] = accuracy
        print(f"max_features={feat_label:>20} → Train Accuracy: {accuracy*100:.2f}%")
    
    return results


def strategy_4_bootstrap_sampling(train_inputs, train_labels):
    """
    Strategy 4: Optimize bootstrap sampling
    
    bootstrap: Whether to use bootstrap samples (default: True)
    max_samples: Control fraction of samples used per tree
    """
    print("\n" + "="*60)
    print("STRATEGY 4: BOOTSTRAP & SAMPLE CONTROL")
    print("="*60)
    
    print("\n--- Testing bootstrap parameter ---")
    for use_bootstrap in [True, False]:
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=10,
            bootstrap=use_bootstrap,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(train_inputs, train_labels)
        accuracy = clf.score(train_inputs, train_labels)
        print(f"bootstrap={use_bootstrap} → Train Accuracy: {accuracy*100:.2f}%")
    
    print("\n--- Testing max_samples (bootstrap ratio) ---")
    max_samples_list = [0.5, 0.7, 0.8, 0.9, 1.0, None]
    results = {}
    
    for max_samp in max_samples_list:
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=10,
            bootstrap=True,
            max_samples=max_samp,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(train_inputs, train_labels)
        accuracy = clf.score(train_inputs, train_labels)
        samp_label = str(max_samp)
        results[samp_label] = accuracy
        print(f"max_samples={samp_label:>4} → Train Accuracy: {accuracy*100:.2f}%")
    
    return results


def strategy_5_criterion_selection(train_inputs, train_labels):
    """
    Strategy 5: Select best splitting criterion
    
    criterion: 'gini' (default) vs 'entropy' vs 'log_loss'
    Different criteria may perform better on different datasets.
    """
    print("\n" + "="*60)
    print("STRATEGY 5: SPLITTING CRITERION SELECTION")
    print("="*60)
    
    criteria = ['gini', 'entropy', 'log_loss']
    results = {}
    
    for criterion in criteria:
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=10,
            criterion=criterion,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(train_inputs, train_labels)
        accuracy = clf.score(train_inputs, train_labels)
        results[criterion] = accuracy
        print(f"criterion='{criterion}' → Train Accuracy: {accuracy*100:.2f}%")
    
    return results


def strategy_6_grid_search(train_inputs, train_labels):
    """
    Strategy 6: Systematic grid search over multiple hyperparameters
    
    Uses GridSearchCV to find optimal combination of parameters.
    This is computationally intensive but thorough.
    """
    print("\n" + "="*60)
    print("STRATEGY 6: GRID SEARCH FOR OPTIMAL PARAMETERS")
    print("="*60)
    
    param_grid = {
        'n_estimators': [300, 500, 700],
        'max_depth': [15, 20, 25],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 4, 8],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=3,  # 3-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("Running grid search (this may take a while)...")
    grid_search.fit(train_inputs, train_labels)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_*100:.2f}%")
    
    # Evaluate on full training set
    best_clf = grid_search.best_estimator_
    train_accuracy = best_clf.score(train_inputs, train_labels)
    print(f"Train Accuracy with best params: {train_accuracy*100:.2f}%")
    
    return best_clf, grid_search.best_params_


def strategy_7_randomized_search(train_inputs, train_labels):
    """
    Strategy 7: Randomized search (faster alternative to grid search)
    
    More efficient for large parameter spaces.
    """
    print("\n" + "="*60)
    print("STRATEGY 7: RANDOMIZED SEARCH")
    print("="*60)
    
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800],
        'max_depth': [10, 15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['gini', 'entropy', 'log_loss']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        rf,
        param_dist,
        n_iter=50,  # Sample 50 random combinations
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("Running randomized search...")
    random_search.fit(train_inputs, train_labels)
    
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV Score: {random_search.best_score_*100:.2f}%")
    
    best_clf = random_search.best_estimator_
    train_accuracy = best_clf.score(train_inputs, train_labels)
    print(f"Train Accuracy with best params: {train_accuracy*100:.2f}%")
    
    return best_clf, random_search.best_params_


def final_tuned_model(train_inputs, train_labels, best_params=None):
    """
    Create final tuned Random Forest with best discovered parameters.
    
    If best_params is provided, use those. Otherwise use recommended defaults
    based on typical good performance for 100-class classification.
    """
    print("\n" + "="*60)
    print("FINAL TUNED RANDOM FOREST MODEL")
    print("="*60)
    
    if best_params is None:
        # Recommended parameters for 100-class classification with 512-d features
        best_params = {
            'n_estimators': 600,          # More trees for complex problem
            'max_depth': 20,               # Reasonable tree depth
            'min_samples_split': 10,       # Prevent shallow splits
            'min_samples_leaf': 4,         # Ensure meaningful leaves
            'max_features': 'sqrt',        # Encourage tree diversity
            'criterion': 'gini',           # Works well for multi-class
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }
    
    print(f"Using parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    clf = RandomForestClassifier(**best_params)
    clf.fit(train_inputs, train_labels)
    
    train_accuracy = clf.score(train_inputs, train_labels)
    print(f"\nFinal Train Accuracy: {train_accuracy*100:.2f}%")
    
    # Feature importance
    feature_importance = clf.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:][::-1]
    print(f"\nTop 10 Most Important Features:")
    for rank, feat_idx in enumerate(top_features_idx, 1):
        print(f"  {rank}. Feature {feat_idx}: {feature_importance[feat_idx]:.4f}")
    
    return clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random Forest Hyperparameter Tuning for 100-class Classification"
    )
    parser.add_argument("--offline_test_path", "-f", type=str, default=None,
                       help="Path to labeled test dataset")
    parser.add_argument("--online_train_path", "-ntr", type=str, default=None,
                       help="Path to training dataset")
    parser.add_argument("--online_test_path", "-nte", type=str, default=None,
                       help="Path to unlabeled test dataset")
    parser.add_argument("--online_save_path", "-nsv", type=str, default=None,
                       help="Path to save predictions")
    parser.add_argument("--model_save_path", "-msv", type=str, default=None,
                       help="Path to save trained model")
    parser.add_argument("--strategy", "-s", type=int, default=6,
                       help="Strategy to run (1-7), default: 6 (grid search)")
    parser.add_argument("--scale", action="store_true",
                       help="Scale features using StandardScaler")
    
    args = parser.parse_args()
    
    # Load training data
    if args.online_train_path is not None:
        print("Loading training data...")
        train_inputs, train_labels = util.load_dataset(
            args.online_train_path, 
            labeled=True
        )
        
        # Scale if requested
        if args.scale:
            print("Scaling features...")
            scaler = StandardScaler()
            train_inputs = scaler.fit_transform(train_inputs).astype(np.float32)
        
        print(f"Dataset shape: {train_inputs.shape}")
        print(f"Number of classes: {len(np.unique(train_labels))}")
        
        # Run selected strategy
        strategy = args.strategy
        
        if strategy == 1:
            best_n_est, results = strategy_1_increase_trees(train_inputs, train_labels)
        elif strategy == 2:
            depth_results, split_results, leaf_results = strategy_2_tree_depth_control(
                train_inputs, train_labels
            )
        elif strategy == 3:
            results = strategy_3_feature_sampling(train_inputs, train_labels)
        elif strategy == 4:
            results = strategy_4_bootstrap_sampling(train_inputs, train_labels)
        elif strategy == 5:
            results = strategy_5_criterion_selection(train_inputs, train_labels)
        elif strategy == 6:
            best_clf, best_params = strategy_6_grid_search(train_inputs, train_labels)
            clf = best_clf
        elif strategy == 7:
            best_clf, best_params = strategy_7_randomized_search(train_inputs, train_labels)
            clf = best_clf
        else:
            print(f"Unknown strategy: {strategy}")
            sys.exit(1)
        
        # Create final model if not already done by strategy
        if strategy not in [6, 7]:
            clf = final_tuned_model(train_inputs, train_labels)
        
        # Offline test if test path provided
        if args.offline_test_path is not None:
            print("\nRunning offline test...")
            util.offline_test(clf, args.offline_test_path, ratio=0.4, scale=args.scale)
        
        # Online test if paths provided
        if (args.online_test_path is not None and 
            args.online_save_path is not None):
            print("\nRunning online test...")
            util.online_test(
                clf,
                train_path=args.online_train_path,
                test_path=args.online_test_path,
                save_path=args.online_save_path
            )
        
        # Save model if path provided
        if args.model_save_path is not None:
            print("\nSaving model...")
            util.save_model(clf, args.model_save_path)
    
    else:
        print("Error: Please provide --online_train_path argument")
        parser.print_help()
