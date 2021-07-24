import pickle
import os
from sklearn import datasets

class IrisModel:
    def __init__(self):
        self.CHECKPOINTS_DIR = "checkpoints/iris"
        self.BEST_IRIS_MODEL = os.path.join(self.CHECKPOINTS_DIR, 'iris_best.pkl')
        self.model = pickle.load(open(self.BEST_IRIS_MODEL,'rb'))
        iris = datasets.load_iris()
        self.labels = iris.target_names

    def predict(self, input_data):
        return [self.labels[i] for i in self.model.predict(input_data)]


iris = IrisModel()
predictions = iris.predict([[4.2,3.1,2.2,1],[3.2,4.4,0.3,2.1]])
print(predictions)