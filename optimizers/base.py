import numpy as np
class Optimizer:
    def set_layers(self, layers):
        raise NotImplementedError()
    
    def update_per_step(self):
        raise NotImplementedError()
    
    def gradient_clip_by_norm(self, grad, max_norm=1.0):
        """Clip gradients by their L2 norm"""
        grad_norm = np.linalg.norm(grad)
        if grad_norm > max_norm:
            grad *= max_norm / grad_norm
        return grad