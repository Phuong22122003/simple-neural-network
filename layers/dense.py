from .base import Layer
from .activation import *
import numpy as np

class Dense(Layer):
    def __init__(self,output_dim,activation = "linear"):
        super().__init__()
        match activation:
            case 'relu':
                self.activation = relu
                self.backward_activation = relu_derivative
            case 'leaky_relu':
                self.activation = leaky_relu
                self.backward_activation = leaky_relu_derivative
            case 'softmax':
                self.activation = softmax
                self.backward_activation = softmax_3d_derivative
            case "linear":
                self.activation = linear
                self.backward_activation = linear_derivative
            case _:
                raise Exception(f"Activation {activation} not found")
            
        self.output_dim = output_dim
        self.weights = None
        self.name = 'Dense'
        
    def init_weight(self,input_dim):
        if self.activation == relu:
            self.weights = np.random.randn(input_dim,self.output_dim) * np.sqrt(2/input_dim)  # He init
        else:
            self.weights = np.random.randn(input_dim,self.output_dim) * np.sqrt(1/input_dim)  # Xavier init
        
        self.bias = np.zeros((1,self.output_dim))
        
    def forward(self, inputs):
        
        input_shape = inputs.shape
        embed_dim = input_shape[-1]
        inputs_flat = inputs.reshape(-1, embed_dim)
    
        if self.weights is None:
            self.init_weight(embed_dim)
    
        self.inputs = inputs_flat
        linear_output = self.inputs @ self.weights + self.bias
        self.linear_outputs = linear_output
        activated = self.activation(linear_output)
         
        return activated.reshape(*input_shape[:-1], -1)
        
    def __call__(self, inputs):
        return self.forward(inputs)
        
    def backward(self, grad_outputs):
        grad_output_shape = grad_outputs.shape
        output_dim = grad_output_shape[-1]
        grad_outputs_flat = grad_outputs.reshape(-1, output_dim)
    
        if self.activation != softmax:
            grad_outputs_flat *= self.backward_activation(self.linear_outputs)
    
        self.grad_weights = self.inputs.T @ grad_outputs_flat
        self.grad_bias = np.sum(grad_outputs_flat, axis=0, keepdims=True)
    
        grad_inputs = grad_outputs_flat @ self.weights.T
        return grad_inputs.reshape(*grad_output_shape[:-1], -1)
    
    def parameters(self):
        return [(self.weights, self.grad_weights), (self.bias, self.grad_bias)]