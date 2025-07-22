import numpy as np
def cross_entropy_derivative(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0] 

def cross_entropy(y_true, y_pred, epsilon=1e-7):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon) # 1e-7 -> 1 - (1e-7)
    loss = -np.sum(y_true * np.log(y_pred), axis=-1)
    return np.mean(loss)