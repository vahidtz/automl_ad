import os
from sklearn.ensemble import RandomForestClassifier
import pickle


def run(model_name, x_train, y_train, x_test, y_test, scoring):
    clf = RandomForestClassifier(n_estimators=200)
    clf = clf.fit(x_train, y_train)

    y_predict = clf.predict(x_test)
    scoring_value = scoring(y_test, y_predict)
    working_directory = os.getcwd()
    filename = working_directory + '/artifacts/' + model_name + '.pickle'
    pickle.dump(clf, open(filename, 'wb'))

    print("Score on test data by Random Forest is:", scoring_value)

    loaded_model = pickle.load(open(filename, 'rb'))
    score = loaded_model.score(x_test, y_test)
    return scoring_value, filename
    print(result)
