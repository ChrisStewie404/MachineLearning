from libsvm.svmutil import *
import numpy as np

def load_libsvm(data_path,return_scipy=True):
    return svm_read_problem(data_path,return_scipy=return_scipy)

def get_accuracy(y_true, y_pred)->float:
    correct_predictions = np.sum(np.array(y_true) == np.array(y_pred))
    total_predictions = len(y_true)
    accuracy = correct_predictions / len(y_true)
    print("Accuracy: {:.2f}% ({}/{})".format(accuracy*100,correct_predictions,total_predictions))
    return accuracy