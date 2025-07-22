from .base import Layer
class Flatten(Layer):
    def forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, grad_outputs):
        return grad_outputs.reshape(self.input_shape)

    def parameters(self):
        return []