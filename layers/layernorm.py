from .base import Layer
from .dense import Dense
from .activation import softmax,softmax_3d_derivative

import numpy as np
class LayerNorm(Layer):
    def __init__(self, embedding_dim, eps=1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.name = 'LayerNorm'
        
        # Learnable parameters
        self.gamma = np.ones((embedding_dim,), dtype=np.float32)  # Scale parameter
        self.beta = np.zeros((embedding_dim,), dtype=np.float32)  # Shift parameter
        
        # Gradients
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)
        
    def forward(self, x):
        """
        x shape: (batch_size, seq_len, embedding_dim)
        """
        self.inputs = x
        
        # Calculate mean and variance along the last dimension (embedding_dim)
        self.mean = np.mean(x, axis=-1, keepdims=True)  # (batch_size, seq_len, 1)
        self.var = np.var(x, axis=-1, keepdims=True)    # (batch_size, seq_len, 1)
        
        # Normalize
        self.x_normalized = (x - self.mean) / np.sqrt(self.var + self.eps)
        
        # Scale and shift
        output = self.gamma * self.x_normalized + self.beta
        
        return output
    
    def __call__(self, x):
        return self.forward(x)
    
    def backward(self, grad_outputs):
        """
        grad_outputs shape: (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, embedding_dim = grad_outputs.shape
        
        # Gradients w.r.t gamma and beta
        self.grad_gamma = np.sum(grad_outputs * self.x_normalized, axis=(0, 1))
        self.grad_beta = np.sum(grad_outputs, axis=(0, 1))
        
        # Gradient w.r.t normalized input
        grad_x_normalized = grad_outputs * self.gamma
        
        # Gradient w.r.t input (complex part - backprop through normalization)
        std_inv = 1.0 / np.sqrt(self.var + self.eps)
        
        # Gradient w.r.t variance
        grad_var = np.sum(grad_x_normalized * (self.inputs - self.mean) * 
                         (-0.5) * std_inv**3, axis=-1, keepdims=True)
        
        # Gradient w.r.t mean
        grad_mean = (np.sum(grad_x_normalized * (-std_inv), axis=-1, keepdims=True) + 
                    grad_var * np.sum(-2.0 * (self.inputs - self.mean), axis=-1, keepdims=True) / embedding_dim)
        
        # Gradient w.r.t input
        grad_input = (grad_x_normalized * std_inv + 
                     grad_var * 2.0 * (self.inputs - self.mean) / embedding_dim + 
                     grad_mean / embedding_dim)
        
        return grad_input
    
    def parameters(self):
        """
        Return trainable parameters and their gradients
        """
        return [(self.gamma, self.grad_gamma), (self.beta, self.grad_beta)]
