import argparse
import numpy as np
import sys
from pathlib import Path

# add repo root so util.py can be imported even when cwd differs
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
import util


class MySVM:
    def __init__(self, learning_rate=0.001, n_iters=1000, C=1.0, delta=1.0, tol=1e-5):
        self.learning_rate_ = learning_rate
        self.n_iters_ = n_iters
        self.C_ = C
        self.delta_ = delta
        self.tol_ = tol
        self.weights_ = None
        self.bias_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        n_samples, n_features = X.shape

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = self.classes_.shape[0]

        self.weights_ = np.zeros((n_features, n_classes), dtype=np.float64)
        self.bias_ = np.zeros(n_classes, dtype=np.float64)

        for _ in range(self.n_iters_):
            scores = X @ self.weights_ + self.bias_
            correct_scores = scores[np.arange(n_samples), y_idx][:, None]
            margins = scores - correct_scores + self.delta_
            margins[np.arange(n_samples), y_idx] = 0.0

            positive_mask = margins > 0.0
            if not np.any(positive_mask):
                break

            coeff = positive_mask.astype(np.float64)
            row_sum = coeff.sum(axis=1)
            coeff[np.arange(n_samples), y_idx] = -row_sum

            # multi-class hinge gradient (Crammer-Singer)
            dw = self.weights_ + (self.C_ / n_samples) * (X.T @ coeff)
            db = (self.C_ / n_samples) * coeff.sum(axis=0)

            max_change = max(
                float(np.max(np.abs(self.learning_rate_ * dw))),
                float(np.max(np.abs(self.learning_rate_ * db)))
            )

            self.weights_ -= self.learning_rate_ * dw
            self.bias_ -= self.learning_rate_ * db

            if max_change < self.tol_:
                break

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = X @ self.weights_ + self.bias_
        indices = np.argmax(scores, axis=1)
        return self.classes_[indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline_test_path", "-f", type=str, default=None)
    parser.add_argument("--online_train_path", "-ntr", type=str, default=None)
    parser.add_argument("--online_test_path", "-nte", type=str, default=None)
    parser.add_argument("--online_save_path", "-nsv", type=str, default=None)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
    parser.add_argument("--n_iters", "-itr", type=int, default=2000)
    parser.add_argument("--C", "-c", type=float, default=1.0)
    parser.add_argument("--delta", "-d", type=float, default=1.0)
    parser.add_argument("--tol", "-t", type=float, default=1e-5)
    args = parser.parse_args()

    clf = MySVM(
        learning_rate=args.learning_rate,
        n_iters=args.n_iters,
        C=args.C,
        delta=args.delta,
        tol=args.tol,
    )

    if args.offline_test_path is not None:
        util.offline_test(clf, args.offline_test_path, ratio=0.4)

    if (
        args.online_train_path is not None
        and args.online_test_path is not None
        and args.online_save_path is not None
    ):
        util.online_test(
            clf,
            train_path=args.online_train_path,
            test_path=args.online_test_path,
            save_path=args.online_save_path,
        )
