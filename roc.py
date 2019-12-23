# This file creates ROC
import pickle
from sklearn.metrics import roc_curve
from matplotlib import pyplot


def run(filename, x_test, y_test, model_name):
    clf = pickle.load(open(filename, 'rb'))

    lr_probs = clf.predict_proba(x_test)
    lr_probs = lr_probs[:, 1]
    ns_probs = [0 for _ in range(y_test.shape[0])]

    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot
    pyplot.figure(2)
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Random Guessing')
    pyplot.plot(lr_fpr, lr_tpr, linestyle='--', label='Positive Class')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the title
    pyplot.title('ROC chart for the best model' + '(' + model_name + ')')
    # show the plot
    pyplot.show()