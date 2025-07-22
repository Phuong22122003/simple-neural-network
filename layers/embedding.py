import numpy as np
from .base import Layer

class Embedding(Layer):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = np.random.randn(vocab_size, embedding_dim) * np.sqrt(1 / embedding_dim)
        self.grad_weights = np.zeros_like(self.weights)
        self.input_indices = None
        self.name = 'Embedding'
        
    def __call__(self,input_indices):
        return self.forward(input_indices)
    def forward(self, input_indices):
        """
        input_indices: (batch_size, seq_len)
        returns: (batch_size, seq_len, embedding_dim)
        """
        self.input_indices = input_indices
        return self.weights[input_indices]  # fancy indexing

    def backward(self, grad_outputs):
        self.grad_weights.fill(0)
        
        # Flatten indices and gradients
        flat_indices = self.input_indices.flatten()
        flat_grads = grad_outputs.reshape(-1, self.embedding_dim)
        
        # Use numpy's add.at for accumulation (equivalent to scatter_add)
        np.add.at(self.grad_weights, flat_indices, flat_grads)
        
        return None


    def parameters(self):
        return [(self.weights, self.grad_weights)]
