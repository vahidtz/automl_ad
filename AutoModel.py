from sklearn.model_selection import train_test_split
import pandas as pd
import scoring
import roc
from sklearn.preprocessing import StandardScaler


class DataModel:
    '''
    inputs:
    filenmae: the file ath of inputs
    scoring type: It should be any of scores mentioned in scoring.py
    test_size: the ratio of test data(between 0 and 1)
    target: target column in the input data
    '''

    def __init__(self, filename, scoring_type, test_size=0.25, target=''):
        self.X = pd.read_csv(filename).drop(target, axis=1)
        self.Y = pd.read_csv(filename)[target]
        self.test_size = test_size
        self.target = target
        self.scoring = scoring.score(scoring_type)
        self.scores_dic = {}
        self.models_files = {}
        self.top_model = ''
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.Y, test_size=self.test_size)
        self.scaler = StandardScaler()
        # self.scaler.fit(self.x_train)
        # self.x_train = self.scaler.transform(self.x_train)
        # self.x_test = self.scaler.transform(self.x_test)

    def run(self, model_name, model):
        return model(model_name=model_name, x_train=self.x_train, y_train=self.y_train, x_test=self.x_test,
                     y_test=self.y_test, scoring=self.scoring)

    # Creating ROC
    def roc(self, best_model_filename, model_name):
        roc.run(best_model_filename, self.x_test, self.y_test, model_name)
