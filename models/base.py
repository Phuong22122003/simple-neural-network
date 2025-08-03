from layers import Layer
from typing import List
from losses import cross_entropy_derivative, cross_entropy
import numpy as np
from tqdm import tqdm

class Model:
    def __init__(self):
        self.optimizer = None
        
    def to_one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]
    
    def compile(self, optimizer):
        self.optimizer = optimizer
        
    
    def fit(self, inputs, outputs, epochs = 1, batch_size=32, to_one_hot = False, verbose = True):
        if self.optimizer == None:
            raise Exception('''optimizer is None
                            def compile(self, optimizer):
                                self.optimizer = optimizer''')
        
        layers = self._all_layer()
        self.optimizer.set_layers(layers)
        for epoch in range(epochs):
            total_loss = 0
            steps_per_epoch = 0
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                
            pbar = tqdm(range(0, inputs.shape[0], batch_size), desc=f"Epoch {epoch+1}", unit="batch") if verbose else range(0, inputs.shape[0], batch_size)
            
            for i in pbar:
                
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
                    
                self.optimizer.update_per_step()
                        
                avg_loss = total_loss / steps_per_epoch
                
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({'loss': avg_loss})
     
    def call(self, inputs, training):
        raise NotImplementedError()
    
    def __call__(self, inputs):
        return self.call(inputs,training=False)
    
    def _all_layer(self):
        layers = []
        for attr in self.__dict__.values():
            if isinstance(attr, Layer):
                layers.append(attr)
        return layers  
        
        