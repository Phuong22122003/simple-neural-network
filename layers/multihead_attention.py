from .base import Layer
from .dense import Dense
from .activation import softmax,softmax_3d_derivative
import numpy as np
class MultiheadAttention(Layer):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head
        self.dropout = dropout
        
        # Weight matrices for Q, K, V projections
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None  # output projection
        
        # Gradients
        self.grad_W_q = None
        self.grad_W_k = None
        self.grad_W_v = None
        self.grad_W_o = None
        
        # Cache for backward pass
        self.Q = None
        self.K = None
        self.V = None
        self.attention_weights = None
        self.inputs = None
        self.context = None
        
        self.name = 'Attention'
        
    def init_weights(self, d_model):
        # Xavier initialization
        scale = np.sqrt(1.0 / d_model)
        
        self.W_q = np.random.randn(d_model, self.d_model) * scale
        self.W_k = np.random.randn(d_model, self.d_model) * scale
        self.W_v = np.random.randn(d_model, self.d_model) * scale
        self.W_o = np.random.randn(self.d_model, self.d_model) * scale
        
        # Initialize gradients
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_W_v = np.zeros_like(self.W_v)
        self.grad_W_o = np.zeros_like(self.W_o)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch_size, num_heads, seq_len, d_k)
        mask: (batch_size, seq_len, seq_len) or (batch_size, 1, seq_len, seq_len)
        """
        d_k = Q.shape[-1]
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        # Apply mask if provided (for preventing attention to padding tokens)
        if mask is not None:
            # Expand mask to match scores dimensions: (batch_size, num_heads, seq_len, seq_len)
            if mask.ndim == 3:  # (batch_size, seq_len, seq_len)
                mask = mask[:, np.newaxis, :, :]  # Add num_heads dimension
            scores = np.where(mask == 0, -1e9, scores)
        
        # Apply softmax
        attention_weights = self.softmax(scores)
        
        # Apply dropout (simplified - just store for backward pass)
        if self.dropout > 0:
            # In practice, you'd apply dropout here
            pass
        
        # Apply attention weights to values
        context = np.matmul(attention_weights, V)
        
        return context, attention_weights
    
    def softmax(self, x):
        """Numerically stable softmax"""
        # Apply along the last dimension (attention over sequence length)
        x_max = np.max(x, axis=-1, keepdims=True)
        x_shifted = x - x_max
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def softmax_derivative(self, softmax_output, grad_output):
        """Vectorized derivative of softmax for backpropagation"""
        # softmax_output: (batch, num_heads, seq_len, seq_len)
        # grad_output: same shape
        
        # Vectorized computation instead of loops
        # grad = softmax * (grad_output - sum(softmax * grad_output, axis=-1, keepdims=True))
        sum_term = np.sum(softmax_output * grad_output, axis=-1, keepdims=True)
        grad_input = softmax_output * (grad_output - sum_term)
        
        return grad_input
    
    def forward(self, inputs, mask=None):
        """
        inputs: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len, seq_len) optional
        """
        batch_size, seq_len, d_model = inputs.shape
        
        # Initialize weights if not done
        if self.W_q is None:
            self.init_weights(d_model)
        
        # Store input for backward pass
        self.inputs = inputs
        
        # Linear projections: (batch_size, seq_len, d_model)
        Q = np.matmul(inputs, self.W_q)
        K = np.matmul(inputs, self.W_k)
        V = np.matmul(inputs, self.W_v)
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Store for backward pass
        self.Q = Q
        self.K = K
        self.V = V
        
        # Apply scaled dot-product attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        self.attention_weights = attention_weights
        
        # Reshape back: (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        self.context = context
        
        # Final linear projection
        output = np.matmul(context, self.W_o)
        
        return output
    
    def backward(self, grad_outputs):
        """
        grad_outputs: (batch_size, seq_len, d_model)
        Optimized for numpy/GPU computation
        """
        batch_size, seq_len, d_model = grad_outputs.shape
        
        # Reshape for efficient computation
        grad_outputs_flat = grad_outputs.reshape(-1, d_model)
        context_flat = self.context.reshape(-1, d_model)
        inputs_flat = self.inputs.reshape(-1, d_model)
        
        # Gradient w.r.t. W_o (vectorized)
        self.grad_W_o = context_flat.T @ grad_outputs_flat
        
        # Gradient w.r.t. context
        grad_context = grad_outputs_flat @ self.W_o.T
        grad_context = grad_context.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Gradient w.r.t. V (batch matrix multiplication)
        grad_V = np.matmul(self.attention_weights.transpose(0, 1, 3, 2), grad_context)
        
        # Gradient w.r.t. attention_weights
        grad_attention_weights = np.matmul(grad_context, self.V.transpose(0, 1, 3, 2))
        
        # Gradient w.r.t. scores (vectorized softmax derivative)
        grad_scores = self.softmax_derivative(self.attention_weights, grad_attention_weights)
        
        # Scale factor
        scale = 1.0 / np.sqrt(self.d_k)
        
        # Gradient w.r.t. Q and K (batch matrix multiplication)
        grad_Q = np.matmul(grad_scores, self.K) * scale
        grad_K = np.matmul(grad_scores.transpose(0, 1, 3, 2), self.Q) * scale
        
        # Reshape back and compute weight gradients efficiently
        grad_Q_flat = grad_Q.transpose(0, 2, 1, 3).reshape(-1, self.d_model)
        grad_K_flat = grad_K.transpose(0, 2, 1, 3).reshape(-1, self.d_model)
        grad_V_flat = grad_V.transpose(0, 2, 1, 3).reshape(-1, self.d_model)
        
        # Weight gradients (vectorized)
        self.grad_W_q = inputs_flat.T @ grad_Q_flat
        self.grad_W_k = inputs_flat.T @ grad_K_flat
        self.grad_W_v = inputs_flat.T @ grad_V_flat
        
        # Gradient w.r.t. inputs
        grad_inputs = ((grad_Q_flat @ self.W_q.T) + 
                      (grad_K_flat @ self.W_k.T) + 
                      (grad_V_flat @ self.W_v.T))
        
        return grad_inputs.reshape(batch_size, seq_len, d_model)
    
    def __call__(self, inputs, mask=None):
        return self.forward(inputs, mask)
    
    def parameters(self):
        return [
            (self.W_q, self.grad_W_q),
            (self.W_k, self.grad_W_k),
            (self.W_v, self.grad_W_v),
            (self.W_o, self.grad_W_o)
        ]