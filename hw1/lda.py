from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from data_loader import *
import time

data_sets = ['svmguide1','cod-rna','gisette_scale']
for data_set in data_sets:
    print(f"Data set: {data_set}")
    y_train, x_train = load_libsvm("./data/"+data_set)
    y_test, x_test = load_libsvm("./data/"+data_set+".t")
    clf = LinearDiscriminantAnalysis()
    start_time = time.time()
    clf.fit(x_train.toarray(),y_train)
    end_time = time.time()
    print("Training time: {:.2f}s".format(end_time-start_time))
    predictions = clf.predict(x_test)
    accuracy = get_accuracy(y_test,predictions)