import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
import pickle


def run(model_name, x_train, y_train, x_test, y_test, scoring):
    clf = Sequential()
    n_features = x_train.shape[1]
    clf.add(Dense(8, input_dim=n_features, kernel_initializer='normal', activation='relu'))
    clf.add(Dense(4, kernel_initializer='normal', activation='relu'))
    clf.add(Dense(1, activation='sigmoid'))
    clf.compile(loss='binary_crossentropy', optimizer='rmsprop')
    history = clf.fit(x_train, y_train,
                      batch_size=100,
                      epochs=30,
                      verbose=2, validation_data=(x_test, y_test)
                      )
    y_predict = clf.predict_classes(x_test)
    scoring_value = scoring(y_test, y_predict)
    working_directory = os.getcwd()
    filename = working_directory + '/artifacts/' + model_name + '.pickle'
    pickle.dump(clf, open(filename, 'wb'))
    print("Score on test data by SVM Tree is:", scoring_value)
    return scoring_value, filename

