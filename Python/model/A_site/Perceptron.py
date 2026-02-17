import numpy as np
import sys
sys.path.append('d:/work/2026/DeepLearning')

from python.model.model import Layer, Model

def step_function(x):
    if x > 0:
        return 1
    return 0

class Perceptron(Model):
    def __init__(self, input_size, output_size, learning_rate=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.layer = PerceptronLayer(input_size, output_size, learning_rate)

    def train(self, X, Y):
        for x, y in zip(X, Y):
            self._train(x.reshape(1,-1), y)

    def predict(self, X):
        ans = []
        for x in X:
            output = self.layer.forward(x.reshape(1,-1))
            ans.append(output)
            print(output)
        print(self.layer)
        return np.array(ans)


    def _train(self, X, Y):
        output =self.layer.forward(X)
        error = Y - output
        self.layer.backward(error)
        self.layer.update(self.learning_rate)
        

class PerceptronLayer(Layer):
    def __init__(self, input_size, output_size, learning_rate):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)

    def forward(self, X):
        self.input = X
        return step_function(np.dot(X, self.weights) + self.bias)

    def backward(self, dY):
        self.dW = np.dot(self.input.T, dY)
        self.db = np.sum(dY, axis=0)

    def update(self, learning_rate):
        self.weights += learning_rate * self.dW
        self.bias += learning_rate * self.db

    def __str__(self):
        return f"PerceptronLayer(input_size={self.input_size}, output_size={self.output_size}, weights={self.weights}, bias={self.bias})"
    

