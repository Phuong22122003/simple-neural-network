from tqdm import tqdm
from typing import List
import numpy as np
from layers import Layer
from losses import cross_entropy,cross_entropy_derivative
from .base import Model


class Model_Adam:
    def __init__(self):
        # Adam optimizer parameters
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Time step
        
    def to_one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]
        
    def gradient_clip_by_norm(self, grad, max_norm=1.0):
        """Clip gradients by their L2 norm"""
        grad_norm = np.linalg.norm(grad)
        if grad_norm > max_norm:
            grad *= max_norm / grad_norm
        return grad
    
    def adam_update(self, param, grad, param_id, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
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
        self.m[param_id] = beta1 * self.m[param_id] + (1 - beta1) * grad
        
        # Update biased second raw moment estimate
        self.v[param_id] = beta2 * self.v[param_id] + (1 - beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m[param_id] / (1 - beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v[param_id] / (1 - beta2 ** self.t)
        
        # Update parameters
        param[:] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
    def fit(self, inputs, outputs, epochs=1, batch_size=32, learning_rate=1e-3, 
            beta1=0.9, beta2=0.999, epsilon=1e-8, verbose=True, to_one_hot = False):
        layers = self._all_layer()
        
        for epoch in range(epochs):
            total_loss = 0
            steps_per_epoch = 0
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")
            pbar = tqdm(range(0, inputs.shape[0], batch_size), desc=f"Epoch {epoch+1}", unit="batch") if verbose else range(0, inputs.shape[0], batch_size)
            
            for i in pbar:
                self.t += 1  # Increment time step for Adam
                
                inputs_batch = inputs[i:i+batch_size, :] 
                if(to_one_hot):
                    outputs_batch = self.to_one_hot(outputs[i:i+batch_size, :], layers[-1].output_dim)
                else:
                    outputs_batch = outputs[i:i+batch_size, :]
                
                # Forward pass
                pred_outputs = self.call(inputs_batch, training=True)
                loss_val = cross_entropy(outputs_batch, pred_outputs)
                
                total_loss += loss_val
                steps_per_epoch += 1
                
                # Backward pass
                dl_da = cross_entropy_derivative(outputs_batch, pred_outputs)
                for layer in reversed(layers):
                    dl_da = layer.backward(dl_da)
                
                # Update parameters using Adam optimizer
                for layer_idx, layer in enumerate(layers):
                    for param_idx, (param, grad) in enumerate(layer.parameters()):
                        # Clip gradients
                        grad_clipped = self.gradient_clip_by_norm(grad)
                        
                        # Create unique parameter ID
                        param_id = f"layer_{layer_idx}_param_{param_idx}"
                        
                        # Apply Adam update
                        self.adam_update(param, grad_clipped, param_id, 
                                       learning_rate, beta1, beta2, epsilon)
                
                avg_loss = total_loss / steps_per_epoch
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({'loss': avg_loss})
     
    def call(self, inputs, training = False):
        raise NotImplementedError()
        
    def __call__(self, inputs):
        return self.call(inputs,training = False)
        
    def _all_layer(self):
        layers = []
        for attr in self.__dict__.values():
            if isinstance(attr, Layer):
                layers.append(attr)
        return layers