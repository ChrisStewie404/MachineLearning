from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import argparse
import os
import sys
from pathlib import Path

# add repo root so util.py can be imported even when cwd differs
REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust parents if nested deeper
sys.path.append(str(REPO_ROOT))
import util
from sklearn.tree import plot_tree

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline_test_path", "-f", type=str, default=None)
    parser.add_argument("--online_train_path", "-ntr", type=str, default=None)
    parser.add_argument("--online_test_path", "-nte", type=str, default=None)
    parser.add_argument("--online_save_path", "-nsv", type=str, default=None)
    parser.add_argument("--models_train_path", "-mtp", type=str, default=None)
    parser.add_argument("--models_test_path", "-mte", type=str, default=None)
    parser.add_argument("--model_save_path", "-msv", type=str, default=None)
    args = parser.parse_args()

    # clf1 = GaussianNB()
    # clf2 = SVC()
    # clf3 = RandomForestClassifier(n_estimators=300)

    # clf1 = KNeighborsClassifier(n_neighbors=10)
    # clf2 = SVC()
    # clf3 = GaussianNB()
    # clf1 = HistGradientBoostingClassifier(
    #     max_iter=1000,
    #     learning_rate=0.05,
    #     max_leaf_nodes=127,
    #     random_state=42)
    # clf2 = SVC()
    # clf3 = GaussianNB()
    # clf = VotingClassifier(
    #     estimators=[('hgb', clf1), ('svm', clf2), ('mlp', clf3)]
        
    # )
    
    # random_states = [24, 42, 2023, 7, 1234, 56, 100, 2024]
    if args.models_train_path is not None and args.models_test_path is not None:
        n_estimators_list = [1900]
        min_sample_leaf = 2
        max_depth = 50
        models = {}
        for n_estimators in n_estimators_list:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                verbose=2,
                max_depth=max_depth,
                n_jobs=-1,
            )
            models[model] = f"../res/RF/pred-rf-{n_estimators}-{max_depth}dp.csv"

        util.online_tests(
            models_paths_map=models,
            train_path=args.models_train_path,
            test_path=args.models_test_path
        )

    if args.offline_test_path is not None:
        clf = RandomForestClassifier(
            n_estimators=1300,
            random_state=42,
            verbose=2,
            n_jobs=-1,
        )
        util.offline_test(clf, args.offline_test_path, ratio=0.4, scale=True)
        print("hyper params: ", clf.get_params())

    if args.online_train_path is not None and args.online_test_path is not None and args.online_save_path is not None:
        clf = RandomForestClassifier(
            n_estimators=1300,
            random_state=42,
            verbose=2,
            n_jobs=-1,
        )
        util.online_test(
            clf, 
            train_path=args.online_train_path, 
            test_path=args.online_test_path,
            save_path=args.online_save_path
        ) 

    if args.model_save_path is not None:
        util.save_model(clf, args.model_save_path)
