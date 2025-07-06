from layers import Dense
from models import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.layer1 = Dense(128,'relu')
        self.layer2 = Dense(10,'softmax')
    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, np.prod(X_train.shape[1:])) / 255.0  # Normalize, shape (60000, 784)
X_test = X_test.reshape(-1, np.prod(X_test.shape[1:])) / 255.0    # Normalize, shape (10000, 784)
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

Y_train_one_hot = np.eye(10)[y_train]  # Shape (60000, 10)
Y_test_one_hot = np.eye(10)[y_test]    # Shape (10000, 10)
model = MyModel()
model.fit(X_train, Y_train_one_hot,epochs=5)