import numpy as np
# def cross_entropy_derivative(y_true, y_pred):
#     epsilon = 1e-9  # Small value to prevent division by zero
#     return - (y_true / (y_pred + epsilon)) / y_true.shape[0]

# def cross_entropy(y_true, y_pred, epsilon=1e-12):
#     """
#     y_true: (batch_size, num_classes) - one-hot
#     y_pred: (batch_size, num_classes) - output of softmax
#     """
#     y_pred = np.clip(y_pred, epsilon, 1. - epsilon) 
#     loss = -np.sum(y_true * np.log(y_pred), axis=1) 
#     return np.mean(loss) 

import numpy as np

def cross_entropy(y_true, y_pred, epsilon=1e-12):
    """
    y_true: (batch_size, num_classes) - one-hot
    y_pred: (batch_size, num_classes) - output of softmax
    """
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon) 
    loss = -np.sum(y_true * np.log(y_pred), axis=1) 
    return np.mean(loss)

def cross_entropy_derivative(y_true, y_pred):
    """
    Gradient of cross-entropy loss when softmax is used as the output activation.
    """
    return (y_pred - y_true) / y_true.shape[0]
