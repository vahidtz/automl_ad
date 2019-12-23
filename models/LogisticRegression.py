import os
import pickle
from sklearn.linear_model import LogisticRegression


def run(model_name, x_train, y_train, x_test, y_test, scoring):

    clf = LogisticRegression(max_iter=100000)
    clf.fit(x_train, y_train)

    y_predict = clf.predict(x_test)
    scoring_value = scoring(y_test, y_predict)
    working_directory = os.getcwd()
    filename = working_directory + '/artifacts/' + model_name + '.pickle'
    pickle.dump(clf, open(filename, 'wb'))

    print("Score on test data by Logistic Regression is:", scoring_value)
    loaded_model = pickle.load(open(filename, 'rb'))
    return scoring_value, filename
    print(result)