import numpy as np
import pandas as pd
import sys
sys.path.append('d:/work/2026/DeepLearning')

from python.test.A_site.dataset.data_Perceptron import load_data
from python.model.A_site.Perceptron import Perceptron


if __name__ == '__main__':

    X,Y = load_data(r'd:\work\2026\DeepLearning\data\csv\and.csv')
    model = Perceptron(2,1,0.5)

    model.train(X,Y)
    predictions = model.predict(X)
    print(predictions)

    print('Accuracy: %.2f' % (np.mean(predictions == Y) * 100))
