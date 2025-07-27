from .base import Model
class Sequential(Model):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def _all_layer(self):
        return self.layers
    def call(self, inputs, training=False):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs