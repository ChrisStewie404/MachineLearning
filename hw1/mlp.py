from sklearn.neural_network import MLPClassifier
from data_loader import *
import time
import numpy as np

def train_test():
    y_train, x_train = load_libsvm("./data/cod-rna")
    y_test, x_test = load_libsvm("./data/cod-rna.t")

    solver = 'lbfgs'
    accuracies = np.zeros((7,9))
    for i in range(2,7):
        for j in range(2,9):
            hidden_layer_sizes = (i,j)
            clf = MLPClassifier(solver=solver,
                                alpha=1e-5,
                                hidden_layer_sizes=hidden_layer_sizes,
                                random_state=1,
                                max_iter=1000)
            start_time = time.time()
            clf.fit(x_train,y_train)
            end_time = time.time()
            print("Hidden layer shape: {}\n Solver: {}\n Training time: {:.2f}s".format(hidden_layer_sizes,solver,end_time-start_time))

            predictions = clf.predict(x_test)
            accuracy = get_accuracy(y_test,predictions)
            accuracies[i,j] = accuracy
    print(accuracies)

if __name__ == "__main__":
    accuracies_result = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0.66666667, 0.88860049, 0.66666667, 0.95386519, 0.88815133, 0.66666667, 0.66666667],
        [0., 0., 0.94935884, 0.66666667, 0.95190286, 0.66666667, 0.95581278, 0.66666667, 0.88805193],
        [0., 0., 0.66666667, 0.66666667, 0.95366269, 0.88669708, 0.88849004, 0.95296318, 0.95178873],
        [0., 0., 0.66666667, 0.94744806, 0.95098613, 0.94770578, 0.94007739, 0.88897602, 0.93062658],
        [0., 0., 0.66666667, 0.76403539, 0.94766528, 0.66666667, 0.94426343, 0.66666667, 0.94857833]
    ])

    data_sets = ['svmguide1','cod-rna','gisette_scale']
    data_set = data_sets[0]
    y_train, x_train = load_libsvm("./data/"+data_set)
    y_test, x_test = load_libsvm("./data/"+data_set+".t")
    results = []
    for i in range(2,17):
        solver = 'lbfgs'
        hidden_layer_sizes = (4,i)
        clf = MLPClassifier(solver=solver,
                            alpha=1e-5,
                            hidden_layer_sizes=hidden_layer_sizes,
                            random_state=1,
                            max_iter=1000)

        start_time = time.time()
        clf.fit(x_train,y_train)
        end_time = time.time()
        print("Hidden layer shape: {}\n Solver: {}\n Training time: {:.2f}s".format(hidden_layer_sizes,solver,end_time-start_time))

        predictions = clf.predict(x_test)
        accuracy = get_accuracy(y_test,predictions)
        results.append("{:.2f}%".format(accuracy*100))
    print(results)