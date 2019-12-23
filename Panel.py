from Models_List import models_dict
from AutoModel import DataModel
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser
#  This is a classification auto ml that runs the skit learn internal different classification models based,
# creates ROC chart and returns the best model along with its score

# reading the parameters
parser = ConfigParser()
parser.read('config.ini')

filename = parser.get('files', 'filename')
test_size = eval(parser.get('model', 'test_size'))
target = parser.get('model', 'target')
scoring_type = parser.get('model', 'scoring_type')

# Creating an instance of experiment and initializing the properties
experiment = DataModel(filename, scoring_type, test_size=test_size, target=target)
results = []
models_list = []
scores_list = []
model_files_list = []

for model_name, model in models_dict.items():
    score, model_file = list(experiment.run(model_name, model))
    results.append({model_name: [score, model_file]})
    models_list.append(model_name)
    scores_list.append(score)
    model_files_list.append(model_file)
print(results)

best_score = max(scores_list)
best_score_index = scores_list.index(best_score)
best_model = models_list[best_score_index]
best_model_file = model_files_list[best_score_index]

# creating bar chart for all classification models
y_pos = np.arange(len(models_list))
scores = scores_list
plt.bar(y_pos, scores_list, align='center', alpha=0.5)
plt.xticks(y_pos, models_list)
plt.ylabel(scoring_type)
plt.title('Model Performance')
plt.show()

# Creating ROC
experiment.roc(best_model_file, best_model)


