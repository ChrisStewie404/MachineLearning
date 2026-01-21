import numpy as np
import argparse
import os
import sys
from pathlib import Path

# add repo root so util.py can be imported even when cwd differs
REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust parents if nested deeper
sys.path.append(str(REPO_ROOT))
import util

class MyLogReg:
    def __init__(self, learning_rate=0.01, n_iters=1000, penalty=None):
        self.learning_rate_ = learning_rate
        self.n_iters_ = n_iters
        self.weights_ = None
        self.bias_ = None
        self.classes_ = None
        self.penalty_ = penalty
        self.eps_ = 1e-6  # convergence threshold
        self.lambda_l2_ = 0.01  
        self.lambda_l1_ = 0.001

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        # Initialize weights: (n_features, n_classes)
        self.weights_ = np.zeros((n_features, n_classes))
        self.bias_ = np.zeros(n_classes)

        # One-hot encode y
        y_encoded = np.zeros((n_samples, n_classes))
        for i, cls in enumerate(self.classes_):
            y_encoded[y == cls, i] = 1

        if self.penalty_ == "l2":
            for _ in range(self.n_iters_):
                linear_model = np.dot(X, self.weights_) + self.bias_
                y_predicted = self._softmax(linear_model)

                # Gradient descent with L2 regularization
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_encoded)) + (self.lambda_l2_ * self.weights_)
                db = (1 / n_samples) * np.sum(y_predicted - y_encoded, axis=0)

                self.weights_ -= self.learning_rate_ * dw
                self.bias_ -= self.learning_rate_ * db

        elif self.penalty_ == "l1":
            for _ in range(self.n_iters_):
                linear_model = np.dot(X, self.weights_) + self.bias_
                y_predicted = self._softmax(linear_model)

                # Gradient descent with L1 regularization
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_encoded)) + self.lambda_l1_ * np.sign(self.weights_)
                db = (1 / n_samples) * np.sum(y_predicted - y_encoded, axis=0)

                self.weights_ -= self.learning_rate_ * dw
                self.bias_ -= self.learning_rate_ * db

        else:
            for _ in range(self.n_iters_):
                linear_model = np.dot(X, self.weights_) + self.bias_
                y_predicted = self._softmax(linear_model)

                # Gradient descent
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_encoded))
                db = (1 / n_samples) * np.sum(y_predicted - y_encoded, axis=0)
                # convergence detection
                if not hasattr(self, "_fit_iter"):
                    self._fit_iter = 0

                max_change = max(np.max(np.abs(self.learning_rate_ * dw)), np.max(np.abs(self.learning_rate_ * db)))
                if max_change < self.eps_:
                    # converged
                    try:
                        delattr(self, "_fit_iter")
                    except Exception:
                        pass
                    break

                self._fit_iter += 1
                if self._fit_iter >= self.n_iters_ - 1:
                    print(f"Warning: training did not converge within {self.n_iters_} iterations.", file=sys.stderr)
                    try:
                        delattr(self, "_fit_iter")
                    except Exception:
                        pass
                self.weights_ -= self.learning_rate_ * dw
                self.bias_ -= self.learning_rate_ * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights_) + self.bias_
        y_predicted = self._softmax(linear_model)
        # Return class with highest probability
        indices = np.argmax(y_predicted, axis=1)
        return self.classes_[indices]

    def _softmax(self, x):
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline_test_path", "-f", type=str, default=None)
    parser.add_argument("--online_train_path", "-ntr", type=str, default=None)
    parser.add_argument("--online_test_path", "-nte", type=str, default=None)
    parser.add_argument("--online_save_path", "-nsv", type=str, default=None)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.01)
    parser.add_argument("--n_iters", "-itr", type=int, default=2000)
    parser.add_argument("--penalty", "-p", type=str, choices=["l1", "l2", "None"], default="None")
    args = parser.parse_args()

    clf = MyLogReg(learning_rate=args.learning_rate, n_iters=args.n_iters, penalty=args.penalty if args.penalty != "None" else None)
    if args.offline_test_path is not None:
        util.offline_test(clf, args.offline_test_path, ratio=0.4)
        print("max: {} min: {} average: {}" .format(np.max(clf.weights_), np.min(clf.weights_), np.mean(clf.weights_)))
    if args.online_train_path is not None and args.online_test_path is not None and args.online_save_path is not None:
        util.online_test(
            clf, 
            train_path=args.online_train_path, 
            test_path=args.online_test_path,
            save_path=args.online_save_path
        )