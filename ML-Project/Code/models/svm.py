from sklearn.svm import SVC,LinearSVC
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
    parser.add_argument("--offline_test_path", "-f", type=str, default=None)
    parser.add_argument("--online_train_path", "-ntr", type=str, default=None)
    parser.add_argument("--online_test_path", "-nte", type=str, default=None)
    parser.add_argument("--online_save_path", "-nsv", type=str, default=None)
    args = parser.parse_args()

    clf = SVC()
    if args.offline_test_path is not None:
        util.offline_test(clf, args.offline_test_path, ratio=0.4)
    if args.online_train_path is not None and args.online_test_path is not None and args.online_save_path is not None:
        util.online_test(
            clf, 
            train_path=args.online_train_path, 
            test_path=args.online_test_path,
            save_path=args.online_save_path
        )

    # clf = LinearSVC()
    # if args.offline_test_path is not None:
    #     util.offline_test(clf, args.offline_test_path, ratio=0.4)
    # if args.online_test_path is not None and args.online_test_path is not None and args.online_save_path is not None:    
    #     util.online_test(
    #             clf, 
    #             train_path=args.online_train_path, 
    #             test_path=args.online_test_path,
    #             save_path=args.online_save_path            
    #     )