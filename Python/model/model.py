class Model:
    def __init__(self):
        pass

    def train(self, X, Y):
        pass

    def predict(self, X):
        pass

    def _train(self, X, Y):
        pass


class Layer:
    def __init__(self):
        pass

    def forward(self, X):
        pass

    def backward(self, dY):
        pass

    def update(self, learning_rate):
        pass

    def __str__(self):
        pass