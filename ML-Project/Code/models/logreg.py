from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import argparse
import os
import sys
from pathlib import Path

# add repo root so util.py can be imported even when cwd differs
REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust parents if nested deeper
sys.path.append(str(REPO_ROOT))
import util

# print learned hyperplanes (coefficients and intercepts)
def _get_coef_intercept(est):
    if hasattr(est, "coef_"):
        return est.coef_, getattr(est, "intercept_", None)
    if hasattr(est, "named_steps"):  # pipeline
        last = list(est.named_steps.values())[-1]
        if hasattr(last, "coef_"):
            return last.coef_, getattr(last, "intercept_", None)
    return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline_test_path", "-f", type=str, default=None)
    parser.add_argument("--online_train_path", "-ntr", type=str, default=None)
    parser.add_argument("--online_test_path", "-nte", type=str, default=None)
    parser.add_argument("--online_save_path", "-nsv", type=str, default=None)
    parser.add_argument("--max_iterations", "-itr", type=int, default=2000)
    parser.add_argument("--penalty", "-p", type=str, choices=["l1", "l2", "elasticnet", "None"], default="l2")
    args = parser.parse_args()

    if args.penalty.lower() == "none":
        args.penalty = None
    clf = OneVsRestClassifier(LogisticRegression(max_iter=args.max_iterations, penalty=args.penalty, solver="liblinear"))
    if args.offline_test_path is not None:
        util.offline_test(clf, args.offline_test_path, ratio=0.4)

        coef, intercept = _get_coef_intercept(clf)
        if coef is not None:
            print("strategy: direct")
            print("coef shape:", coef.shape)
            print("coef:", coef)
            print("intercept:", intercept)
        else:
            print("strategy: one-vs-rest estimators")
            for i, est in enumerate(getattr(clf, "estimators_", [])):
                c, b = _get_coef_intercept(est)
                if c is not None:
                    print(f"class {i} coef shape: {c.shape}")
                    print(f"class {i} coef: {c}")
                    print(f"class {i} intercept: {b}")
                else:
                    print(f"class {i}: no coef_/intercept found")
    if args.online_train_path is not None and args.online_test_path is not None and args.online_save_path is not None:
        util.online_test(
            clf, 
            train_path=args.online_train_path, 
            test_path=args.online_test_path,
            save_path=args.online_save_path
        )    