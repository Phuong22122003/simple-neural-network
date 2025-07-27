from .base import Layer
from .dense import Dense
from .activation import softmax,softmax_3d_derivative
from .multihead_attention import MultiheadAttention
from .layernorm import LayerNorm
import numpy as np
class TransformerBlock(Layer):
    """Transformer block with proper residual connections and backward pass"""
    
    def __init__(self, embedding_dim, n_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        # Sub-layers
        self.attention = MultiheadAttention(embedding_dim,n_heads)
        self.ffn1 = Dense(d_ff, activation='relu')
        self.ffn2 = Dense(embedding_dim, activation='linear')
        
        # Layer normalization
        self.ln1 = LayerNorm(embedding_dim)  # Before attention
        self.ln2 = LayerNorm(embedding_dim)  # Before FFN
        
        # Cache for backward pass
        self.input_cache = None
        self.ln1_output_cache = None
        self.attn_output_cache = None
        self.attn_residual_cache = None
        self.ln2_output_cache = None
        self.ffn1_output_cache = None
        self.ffn2_output_cache = None
        
        self.name = 'TransformerBlock'
    
    def forward(self, x, mask=None):
        """
        Proper transformer architecture with residual connections:
        1. x = x + MultiHeadAttention(LayerNorm(x))
        2. x = x + FFN(LayerNorm(x))
        """
        # Cache input for backward
        self.input_cache = x.copy()
        
        # === Multi-Head Attention với Residual ===
        # 1. Layer norm before attention
        ln1_output = self.ln1.forward(x)
        self.ln1_output_cache = ln1_output.copy()
        
        # 2. Multi-head attention
        attn_output = self.attention.forward(ln1_output, mask=mask)
        self.attn_output_cache = attn_output.copy()
        
        # 3. Residual connection cho attention
        x = x + attn_output  # Shortcut connection
        self.attn_residual_cache = x.copy()
        
        # === Feed-Forward Network with Residual ===
        # 4. Layer norm before FFN
        ln2_output = self.ln2.forward(x)
        self.ln2_output_cache = ln2_output.copy()
        
        # 5. First FFN layer (Linear + ReLU)
        ffn1_output = self.ffn1.forward(ln2_output)
        # ffn1_output = self.ffn1.forward(ln2_output, mask=mask)
        self.ffn1_output_cache = ffn1_output.copy()
        
        # 6. Second FFN layer (Linear)
        ffn2_output = self.ffn2.forward(ffn1_output)
        # ffn2_output = self.ffn2.forward(ffn1_output, mask=mask)
        self.ffn2_output_cache = ffn2_output.copy()
        
        # 7. Residual connection cho FFN
        x = x + ffn2_output  # Shortcut connection
        
        return x
    
    def backward(self, grad_output):
        """
        Backward pass with residual connections
        
        Architecture:
        x₁ = x₀ + attention(ln1(x₀))
        x₂ = x₁ + ffn(ln2(x₁))
        
        Chain rule:
        ∂L/∂x₀ = ∂L/∂x₂ × [∂x₂/∂x₁ + ∂x₂/∂ffn × ∂ffn/∂ln2 × ∂ln2/∂x₁] × [∂x₁/∂x₀ + ∂x₁/∂attn × ∂attn/∂ln1 × ∂ln1/∂x₀]
        """
        
        # === Backward throught FFN Residual ===
        # grad_output = ∂L/∂x₂ = ∂L/∂(x₁ + ffn2_output)
        
        # 1. Gradient from shortcut path: ∂(x₁ + ffn)/∂x₁ = 1
        grad_ffn_shortcut = grad_output.copy()
        
        # 2. Gradient throught FFN path
        # Backward throught ffn2
        grad_ffn1_output = self.ffn2.backward(grad_output)
        
        # Backward throught ffn1  
        grad_ln2_output = self.ffn1.backward(grad_ffn1_output)
        
        # Backward throught ln2
        grad_attn_residual = self.ln2.backward(grad_ln2_output)
        
        # 3. Tổng gradient at x₁ = attn_residual_cache
        grad_x1 = grad_ffn_shortcut + grad_attn_residual
        
        # === Backward throught Attention Residual ===
        # grad_x1 = ∂L/∂x₁ = ∂L/∂(x₀ + attn_output)
        
        # 4. Gradient from shortcut path: ∂(x₀ + attn)/∂x₀ = 1  
        grad_attn_shortcut = grad_x1.copy()
        
        # 5. Gradient throught Attention path
        # Backward throught attention
        grad_ln1_output = self.attention.backward(grad_x1)
        
        # Backward throught ln1
        grad_input = self.ln1.backward(grad_ln1_output)
        
        # 6. Sum gradient at x₀ = input_cache
        grad_x0 = grad_attn_shortcut + grad_input
        
        return grad_x0
    
    def parameters(self):
        """Collect all parameters from sub-layers for updating"""
        params = []
        
        # Attention parameters
        params.extend(self.attention.parameters())
        
        # FFN parameters  
        params.extend(self.ffn1.parameters())
        params.extend(self.ffn2.parameters())
        
        # LayerNorm parameters
        params.extend(self.ln1.parameters())
        params.extend(self.ln2.parameters())
        
        return params
    
    def dropout(self, x, training=True):
        """Dropout regularization (optional)"""
        if not training or self.dropout_rate == 0:
            return x
        mask = np.random.binomial(1, 1-self.dropout_rate, x.shape) / (1-self.dropout_rate)
        return x * mask