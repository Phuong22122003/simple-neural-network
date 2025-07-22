from .base import Layer
from .dense import Dense
from .activation import softmax,softmax_3d_derivative
import numpy as np

class Attention(Layer):
    def __init__(self,embedding_dim):
        self.wq = Dense(embedding_dim)
        self.wk = Dense(embedding_dim)
        self.wv = Dense(embedding_dim)
        self.name = 'Attention'
    def forward(self,inputs,mask=None):
        
        embedding_dim = inputs.shape[-1]
        
        self.embedding_dim = embedding_dim
        
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        
        score = (q @ k.transpose(0,2,1) / np.sqrt(embedding_dim))
        
        if mask is not None:
            score = np.where(mask==0, -1e9, score)
            
        score = softmax(score)

        self.mask = mask
        self.score = score
        self.q = q
        self.k = k
        self.v = v
        
        return score @ v

    def __call__(self,inputs,mark=None):
        return self.forward(inputs,mark)
         
    def backward(self, grad_outputs):
        # grad_outputs shape: (batch, seq_len, embed_dim)
        
        # 1. ∂L/∂score
        dl_dscore = grad_outputs @ self.v.transpose(0, 2, 1)  # (batch, seq_len, seq_len)
    
        # 2. ∂softmax/∂score (Jacobian, simplified)

        dscore_draw  = softmax_3d_derivative(self.score)
        dl_draw = np.einsum("btil,bil->bti", dscore_draw, dl_dscore)
        # dl_draw = np.where(self.mask==0,0,dl_draw)

        # 3. ∂L/∂Q
        d_q = dl_draw @ self.k / np.sqrt(self.embedding_dim) # (batch, seq_len, embed_dim)
        grad_inputs_q = self.wq.backward(d_q)
    
        # 4. ∂L/∂K
        d_k = dl_draw.transpose(0, 2, 1) @ self.q / np.sqrt(self.embedding_dim)
        grad_inputs_k = self.wk.backward(d_k)
    
        # 5. ∂L/∂V
        d_v = self.score.transpose(0, 2, 1) @ grad_outputs
        grad_inputs_v = self.wv.backward(d_v)
    
        grad_inputs = grad_inputs_q + grad_inputs_k + grad_inputs_v
        return grad_inputs

    def parameters(self):
        return self.wq.parameters() + self.wk.parameters() + self.wv.parameters()