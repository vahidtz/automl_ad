# k-fold by SVM with a linear kernel
import os
from sklearn import svm
import pickle
from sklearn.model_selection import cross_val_score


def run(model_name, x_train, y_train, x_test, y_test, scoring):
    C = 0.9
    clf = svm.SVC(kernel='linear', C=C, probability=True)
    clf.fit(x_train, y_train)

    y_predict = clf.predict(x_test)
    scoring_value = scoring(y_test, y_predict)
    working_directory = os.getcwd()
    filename = working_directory + '/artifacts/' + model_name + '.pickle'
    pickle.dump(clf, open(filename, 'wb'))

    print("Score on test data by SVM is:", scoring_value)
    loaded_model = pickle.load(open(filename, 'rb'))
    return scoring_value, filename
    print(result)
