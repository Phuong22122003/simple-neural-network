import numpy as np
from .base import Optimizer
class Adam(Optimizer):
    def __init__(self, learning_rate = 1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m = {}
        self.v = {}
        self.t = 0 
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.layers = None

    def set_layers(self, layers):
        self.layers = layers
        
    def update_param(self, param, grad, param_id):
        """
        Adam optimizer update
        Args:
            param: parameter to update
            grad: gradient
            param_id: unique identifier for parameter
            learning_rate: learning rate (alpha)
            beta1: exponential decay rate for first moment estimates
            beta2: exponential decay rate for second moment estimates
            epsilon: small constant for numerical stability
        """
        # Initialize first and second moment estimates if not exists
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(param)
            self.v[param_id] = np.zeros_like(param)
        
        # Update biased first moment estimate
        self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
        
        # Update biased second raw moment estimate
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
        
        # Update parameters
        param[:] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
    def update_per_step(self):
        self.t += 1
        for layer_idx, layer in enumerate(self.layers):
            for param_idx, (param, grad) in enumerate(layer.parameters()):
                # Clip gradients
                grad_clipped = self.gradient_clip_by_norm(grad)
                
                # Create unique parameter ID
                param_id = f"layer_{layer_idx}_param_{param_idx}"
                
                # Apply Adam update
                self.update_param(param, grad_clipped, param_id)