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
    expZ = np.exp(Z - np.max(Z, axis=-1, keepdims=True)) 
    return expZ / np.sum(expZ, axis=-1, keepdims=True)

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

def softmax_3d_derivative_cpu(S):
    
    B, N, L = S.shape
    grads = np.zeros((B, N, L, L))
    
    for b in range(B):
        for n in range(N):
            s = S[b, n].reshape(-1, 1)  # shape (L, 1)
            grads[b, n] = np.diagflat(s) - s @ s.T
    
    return grads

def softmax_3d_derivative(S):
    """
    Ultra-fast version using einsum for maximum efficiency
    """
    B, N, L = S.shape
    
    # Create identity matrix
    I = np.eye(L)
    
    # Vectorized computation using einsum
    # S[:, :, :, None] * I[None, None, :, :] creates diagonal matrices
    # S[:, :, :, None] * S[:, :, None, :] creates outer products
    grads = (S[:, :, :, None] * I[None, None, :, :] - 
             S[:, :, :, None] * S[:, :, None, :])
    
    return grads