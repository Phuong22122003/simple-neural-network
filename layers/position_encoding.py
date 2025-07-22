import math
from .base import Layer
from .dense import Dense
from .activation import softmax,softmax_3d_derivative

import numpy as np
class PositionalEncoding(Layer):
    def __init__(self, max_len, embedding_dim):
        super().__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.name = 'PositionalEncoding'
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, embedding_dim))
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        
        # Create div_term for sinusoidal encoding
        div_term = np.exp(np.arange(0, embedding_dim, 2, dtype=np.float32) * 
                         -(math.log(10000.0) / embedding_dim))
        
        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cos to odd indices (handle odd embedding_dim case)
        if embedding_dim % 2 == 0:
            pe[:, 1::2] = np.cos(position * div_term)
        else:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        
        # Store as parameter (non-trainable)
        self.pe = pe
        
    def forward(self, x, training = False):
        """
        x shape: (batch_size, seq_len, embedding_dim)
        """

        batch_size, seq_len, embedding_dim = x.shape
        
        if training == False:
            # Create positional encoding matrix
            pe = np.zeros((seq_len, embedding_dim))
            position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)
            
            # Create div_term for sinusoidal encoding
            div_term = np.exp(np.arange(0, embedding_dim, 2, dtype=np.float32) * 
                             -(math.log(10000.0) / embedding_dim))
            
            pe[:, 0::2] = np.sin(position * div_term)
            
            if embedding_dim % 2 == 0:
                pe[:, 1::2] = np.cos(position * div_term)
            else:
                pe[:, 1::2] = np.cos(position * div_term[:-1])
            
            pos_encoding = pe[:seq_len, :].reshape(1, seq_len, embedding_dim)
            

            return x + pos_encoding

        
        # Add positional encoding to input
        # pe[:seq_len] selects the first seq_len positions
        pos_encoding = self.pe[:seq_len, :].reshape(1, seq_len, embedding_dim)
        
        # Store input for backward pass
        self.inputs = x
        
        return x + pos_encoding
    
    def __call__(self, x,training=False):
        return self.forward(x,training=training)
    
    def backward(self, grad_outputs):
        """
        Positional encoding doesn't have trainable parameters,
        so gradient just passes through unchanged
        """
        return grad_outputs
    
    def parameters(self):
        """
        No trainable parameters for positional encoding
        """
        return []