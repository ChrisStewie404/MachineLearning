from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score
import os
import sys
from pathlib import Path

# add repo root so util.py can be imported even when cwd differs
REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust parents if nested deeper
sys.path.append(str(REPO_ROOT))
import util
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", "-r", type=float, default=0.4)
    args = parser.parse_args()
    ratio = args.train_ratio

    X, y = util.load_dataset('../data/train.csv', labeled=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio, random_state=42, shuffle=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # LR tuning
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000))
    ])
    lr_grid = {
        'lr__C': [0.1, 1, 10, 100],
        'lr__class_weight': [None, 'balanced']
    }
    lr_search = GridSearchCV(lr_pipe, lr_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    lr_search.fit(X_train, y_train)

    # SVM tuning
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True))
    ])
    svm_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 1e-3],
        'svm__class_weight': [None, 'balanced']
    }
    svm_search = GridSearchCV(svm_pipe, svm_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    svm_search.fit(X_train, y_train)

    # Compare on test set
    lr_pred = lr_search.best_estimator_.predict(X_test)
    svm_pred = svm_search.best_estimator_.predict(X_test)

    print(f"LR macro-F1: {f1_score(y_test, lr_pred, average='macro'):.4f}, params: {lr_search.best_params_}")
    print(f"SVM macro-F1: {f1_score(y_test, svm_pred, average='macro'):.4f}, params: {svm_search.best_params_}")