from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score


def score(scoring):
    scoring_dic = {"accuracy": accuracy_score, "precision": average_precision_score,
                   "f1_score": f1_score, "log_loss": log_loss, "recall": recall_score}
    return scoring_dic[scoring]
