from sklearn.linear_model import LogisticRegression
from data_loader import load_libsvm,get_accuracy
import numpy as np

data_sets = ['svmguide1','cod-rna','gisette_scale']
for data_set in data_sets:
    print(f"Data set: {data_set}")
    y_train, x_train = load_libsvm("./data/"+data_set)
    y_test, x_test = load_libsvm("./data/"+data_set+".t")

    clf = LogisticRegression(random_state=0,max_iter=500).fit(x_train, y_train)
    predictions = clf.predict(x_test)
    accuracy = get_accuracy(y_test, predictions)


