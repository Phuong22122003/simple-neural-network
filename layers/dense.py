from .base import Layer
from .activation import *
import numpy as np
class Dense(Layer):
    def __init__(self,output_dim,activation = None):
        super().__init__()
        if activation == 'relu':
            self.activation = relu
            self.backward_activation = relu_derivative
        elif activation == 'leaky_relu':
            self.activation = leaky_relu
            self.backward_activation = leaky_relu_derivative
        elif activation == 'softmax':
            self.activation = softmax
            self.backward_activation = softmax_derivative_from_output_2d
        else:
            self.activation = self.linear
            self.backward_activation = self.linear_derivative
            
        self.output_dim = output_dim
        self.weights = None
    def init_weight(self,input_dim):
        self.weights = np.random.randn(input_dim,self.output_dim)* np.sqrt(2/input_dim)
        self.bias = np.zeros((1,self.output_dim))
    def forward(self, inputs):
        '''
        inputs shape(m,n)
        weight shape(n,k)
        outputs shape(m,k)
        '''

        if self.weights is None:
            input_dim = inputs.shape[1]
            self.init_weight(input_dim=input_dim)
        self.inputs = inputs
        self.linear_outputs = self.inputs @ self.weights + self.bias
        self.outputs = self.activation(self.linear_outputs)
        return self.outputs
    def __call__(self, inputs):
        return self.forward(inputs)
    def backward(self, grad_outputs):
        
        batch_size = grad_outputs.shape[0]
        if self.activation == softmax:
            pass
            # # For softmax, we need to handle the Jacobian properly
            # da_dz = self.backward_activation(self.outputs)  # (batch_size, output_dim, output_dim)
            # dl_dz = np.zeros_like(grad_outputs)  # (batch_size, output_dim)
            
            # for i in range(batch_size):
            #     dl_dz[i, :] = da_dz[i] @ grad_outputs[i, :]
            # grad_outputs = dl_dz
        else:
            # For other activations
            grad_outputs = grad_outputs * self.backward_activation(self.linear_outputs)  # (batch_size, output_dim)

        # Calculate gradients
        self.grad_weights = self.inputs.T @ grad_outputs  # (input_dim, output_dim)
        self.grad_bias = np.sum(grad_outputs, axis=0, keepdims=True)  # (1, output_dim)
        # Calculate gradient for previous layer
        grad_inputs = grad_outputs @ self.weights.T  # (batch_size, input_dim) because dl2 = da1 @ weight => dl2/da  = weight.T
        return grad_inputs
    
    def parameters(self):
        return [(self.weights, self.grad_weights), (self.bias, self.grad_bias)]
    def update_weights(self,new_weights,new_bias):
        self.weights = new_weights
        self.bias = new_bias