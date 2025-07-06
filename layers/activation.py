import numpy as np
def linear(Z):
    return Z

def linear_derivative(Z):
    return np.ones_like(Z)


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return np.where(Z > 0, 1, 0)


def leaky_relu(Z, alpha=0.01):
    return np.where(Z > 0, Z, alpha * Z)


def leaky_relu_derivative(Z, alpha=0.01):
    return np.where(Z > 0, 1, alpha)


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True)) 
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(x, grad_output):
    sig = sigmoid(x)
    return grad_output * sig * (1 - sig)

def softmax_derivative(s): 
    """
    s: shape=(num_class)
    """
    s = s.reshape(-1, 1)  # convert to vector column
    return np.diagflat(s) - np.dot(s, s.T)  # Jacobian matrix

def softmax_derivative_from_output_2d(softmax_output):
    """
    Calculate Jacobian of Softmax for each sample in batch.
    
    softmax_output: numpy array (shape: [batch_size, num_classes])
    Output: numpy array (shape: [batch_size, num_classes, num_classes])
    """
    batch_size, num_classes = softmax_output.shape
    jacobian_matrices = np.zeros((batch_size, num_classes, num_classes))

    for i in range(batch_size): 
        s = softmax_output[i, :].reshape(-1, 1) 
        jacobian_matrices[i] = np.diagflat(s) - np.dot(s, s.T)

    return jacobian_matrices