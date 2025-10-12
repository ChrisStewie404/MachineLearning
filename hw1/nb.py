from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from data_loader import load_libsvm,get_accuracy

data_sets = ['svmguide1','cod-rna','gisette_scale']
for data_set in data_sets:
    print(f"Data set: {data_set}")
    y_train, x_train = load_libsvm("./data/"+data_set)
    y_test, x_test = load_libsvm("./data/"+data_set+".t")

    clf = GaussianNB()
    clf.fit(x_train.toarray(), y_train)
    predictions = clf.predict(x_test.toarray())
    print("GaussianNB:")
    accuracy = get_accuracy(y_test, predictions)

    clf = BernoulliNB()
    clf.fit(x_train.toarray(), y_train)
    predictions = clf.predict(x_test.toarray())
    print("BernoulliNB:")
    accuracy = get_accuracy(y_test, predictions)