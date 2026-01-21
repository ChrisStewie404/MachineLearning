from sklearn import tree
import argparse
import os
import sys
from pathlib import Path

# add repo root so util.py can be imported even when cwd differs
REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust parents if nested deeper
sys.path.append(str(REPO_ROOT))
import util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--criterions","-c", type=str, default="gini", help="Split criteria (gini, entropy, log_loss)")
    parser.add_argument("--max_depth", "-d", type=int, default=None, help="The maximum depth of the tree")
    parser.add_argument("--min_samples_leaf", type=int, default=1, help="The minimum number of samples required to be at a leaf node")
    parser.add_argument("--ccp_alpha", type=float, default=0.0, help="Complexity parameter used for Minimal Cost-Complexity Pruning")
    args = parser.parse_args()

    criterions = args.criterions.split(',')

    # offline test
    for criterion in criterions:
        print(f"Testing Decision Tree with criterion: {criterion}, max_depth: {args.max_depth}, min_samples_leaf: {args.min_samples_leaf}, ccp_alpha: {args.ccp_alpha}")
        clf = tree.DecisionTreeClassifier(
            criterion=criterion,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            ccp_alpha=args.ccp_alpha
        )
        util.offline_test(clf, '../data/train.csv', ratio=0.4)

    # # online test
    # for criterion in criterions:
    #     clf = tree.DecisionTreeClassifier(
    #         criterion=criterion,
    #         max_depth=args.max_depth,
    #         min_samples_leaf=args.min_samples_leaf,
    #         ccp_alpha=args.ccp_alpha
    #     )
    #     util.online_test(clf, save_path=f'./res/DT/pred-dt-{criterion}.csv')