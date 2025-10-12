from libsvm.svmutil import *
import numpy as np

import argparse

def load_libsvm(data_path,return_scipy=True):
    return svm_read_problem(data_path,return_scipy=return_scipy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=bool, default=False, help='whether to scale the data')
    args = parser.parse_args()

data_sets = ['svmguide1','cod-rna','gisette_scale']
for data_set in data_sets:
    print(f"Data set: {data_set}")
    y_train, x_train = load_libsvm("./data/"+data_set)
    y_test, x_test = load_libsvm("./data/"+data_set+".t")

    if args.scale:
        scale_param = csr_find_scale_param(x_train, lower=-1,upper=1)
        x_train = csr_scale(x_train, scale_param)
        x_test = csr_scale(x_test, scale_param)

    m = svm_train(y_train, x_train, '-q')
    pred_labels, pred_metrics, pred_values = svm_predict(y_test, x_test, m)