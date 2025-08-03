from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, learning_rate = 1e-1):
        super().__init__()
        self.learning_rate = learning_rate
    def update_per_step(self):
        for layer in self.layers:
            for param, grad in layer.parameters():
                grad_clipped  =  self.gradient_clip_by_norm(grad)
                param[:]-= self.learning_rate * grad_clipped
    def set_layers(self, layers):
        self.layers = layers