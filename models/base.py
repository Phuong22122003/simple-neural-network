from layers import Layer
from typing import List
from losses import cross_entropy_derivative,cross_entropy
import numpy as np
class Model:
    def __init__(self):
        self.layers:List[Layer] = []
        pass
    def fit(self, inputs, outputs, epochs = 1, batch_size=32, learning_rate=1e-4, loss=None):
        self.layers = self._all_layer()
        steps = inputs.shape[0]// batch_size
        for epoch in range(epochs):
            total_loss = 0
            steps_per_epoch = 0
            for i in range(0, inputs.shape[0], batch_size):
                inputs_batch = inputs[i:i+batch_size, :] 
                outputs_batch = outputs[i:i+batch_size, :]   
                
                # Forward pass
                pred_outputs = inputs_batch
                for layer in self.layers:
                    pred_outputs = layer.forward(pred_outputs)
                loss_val = cross_entropy(outputs_batch, pred_outputs)
                total_loss += loss_val
                steps_per_epoch += 1
                # Backward pass
                dl_da = cross_entropy_derivative(outputs_batch, pred_outputs)
                for layer in reversed(self.layers):
                    dl_da = layer.backward(dl_da)
                for layer in self.layers:
                    i = 0
                    for param, grad in layer.parameters():
                        np.clip(grad, -1, 1, out=grad)
                        param[:]-= learning_rate * grad
 
            avg_loss = total_loss / steps
            print(f'Epoch: {epoch+1}/{epochs} ==========================> {steps}/{steps} steps. loss {avg_loss}')

     
    def call(self,inputs):
        raise NotImplementedError()
    def __call__(self, inputs):
        return self.call(inputs)
    def _all_layer(self):
        layers = []
        for attr in self.__dict__.values():
            if isinstance(attr,Layer):
                layers.append(attr)
        return layers  
        