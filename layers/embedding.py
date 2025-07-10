import numpy as np
from .base import Layer

class Embedding(Layer):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = np.random.randn(vocab_size, embedding_dim) * np.sqrt(1 / vocab_size)
        self.grad_weights = np.zeros_like(self.weights)
        self.input_indices = None

    def forward(self, input_indices):
        """
        input_indices: (batch_size, seq_len)
        returns: (batch_size, seq_len, embedding_dim)
        """
        self.input_indices = input_indices
        return self.weights[input_indices]  # fancy indexing

    def backward(self, grad_outputs):
        """
        grad_outputs: (batch_size, seq_len, embedding_dim)
        → Tính gradient cho mỗi chỉ số input, cộng dồn vào self.grad_weights
        """
        self.grad_weights.fill(0)  # reset gradient

        # Cộng dồn gradient theo từng chỉ số
        batch_size, seq_len = self.input_indices.shape
        for i in range(batch_size):
            for j in range(seq_len):
                idx = self.input_indices[i, j]
                self.grad_weights[idx] += grad_outputs[i, j]

        return None  # không truyền ngược gradient vì input là số nguyên

    def parameters(self):
        return [(self.weights, self.grad_weights)]

    def update_weights(self, new_weights, _=None):  # không có bias
        self.weights = new_weights
