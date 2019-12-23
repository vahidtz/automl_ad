from models import DecisionTree
from models import RandomForest
from models import SVM
from models import KNN
from models import LogisticRegression
from models import NN
models_dict = {"Decision Tree": DecisionTree.run, "Random Forest": RandomForest.run, "SVM": SVM.run,
               "KNN": KNN.run, "Logistic Regression": LogisticRegression.run, "Neural Network": NN.run}



