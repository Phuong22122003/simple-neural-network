import numpy as np

class Dense:
    @staticmethod
    def linear(Z):
        return Z

    @staticmethod
    def linear_derivative(Z):
        return np.ones_like(Z)
    
    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z):
        return np.where(Z > 0, 1, 0)
    
    @staticmethod
    def leaky_relu(Z, alpha=0.01):
        return np.where(Z > 0, Z, alpha * Z)

    @staticmethod
    def leaky_relu_derivative(Z, alpha=0.01):
        return np.where(Z > 0, 1, alpha)
    
    @staticmethod
    def softmax(Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True)) 
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_derivative(s): 
        """
        s: shape=(num_class)
        """
        s = s.reshape(-1, 1)  # convert to vector column
        return np.diagflat(s) - np.dot(s, s.T)  # Jacobian matrix
    
    @staticmethod
    def softmax_derivative_from_output_2d(softmax_output):
        """
        Calculate Jacobian of Softmax for each sample in batch.
        
        softmax_output: numpy array (shape: [batch_size, num_classes])
        Output: numpy array (shape: [batch_size, num_classes, num_classes])
        """
        batch_size, num_classes = softmax_output.shape
        jacobian_matrices = np.zeros((batch_size, num_classes, num_classes))

        for i in range(batch_size): 
            s = softmax_output[i, :].reshape(-1, 1) 
            jacobian_matrices[i] = np.diagflat(s) - np.dot(s, s.T)

        return jacobian_matrices
    
    def __init__(self, output_size, activation='relu', learning_rate=1e-4):
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.is_output = False
        self.weight = None
        self.bias = None
        
        if activation == 'relu':
            self.activation = self.relu
            self.backward_activation = self.relu_derivative
        elif activation == 'leaky_relu':
            self.activation = self.leaky_relu
            self.backward_activation = self.leaky_relu_derivative
        elif activation == 'softmax':
            self.activation = self.softmax
            self.backward_activation = self.softmax_derivative_from_output_2d
        else:
            self.activation = self.linear
            self.backward_activation = self.linear_derivative

    def get_input_size(self):
        return self.input_size
    
    def get_output_size(self):
        return self.output_size

    def init_weight(self, input_size):
        self.input_size = input_size
        # Xavier initialization
        self.weight = np.random.randn(input_size, self.output_size) * np.sqrt(2/input_size)
        self.bias = np.zeros((1, self.output_size))

    def forward(self, inputs):
        # inputs.shape = (batch_size, input_size)
        # weight.shape = (input_size, output_size)
        
        if self.weight is None:
            self.init_weight(inputs.shape[1])
        
        # Z.shape = (batch_size, output_size)
        Z = inputs @ self.weight + self.bias
        
        # A.shape = (batch_size, output_size)
        A = self.activation(Z)
        
        # saving for backward
        self.inputs = inputs 
        self.Z = Z
        self.A = A
        
        return A
    
    def backward(self, dl_da):
        # dl_da.shape = (batch_size, output_size)
        batch_size = dl_da.shape[0]
        
        if self.activation == self.softmax:
            # For softmax, we need to handle the Jacobian properly
            da_dz = self.backward_activation(self.A)  # (batch_size, output_size, output_size)
            dl_dz = np.zeros_like(dl_da)  # (batch_size, output_size)
            
            for i in range(batch_size):
                dl_dz[i, :] = da_dz[i] @ dl_da[i, :]
        else:
            # For other activations
            dl_dz = dl_da * self.backward_activation(self.Z)  # (batch_size, output_size)

        # Calculate gradients
        dl_dw = self.inputs.T @ dl_dz  # (input_size, output_size)
        dl_db = np.sum(dl_dz, axis=0, keepdims=True)  # (1, output_size)
        
        # Update weights and bias
        self.weight = self.weight - self.learning_rate * dl_dw
        self.bias = self.bias - self.learning_rate * dl_db
        
        # Calculate gradient for previous layer
        pre_dl_da = dl_dz @ self.weight.T  # (batch_size, input_size) because dl2 = da1 @ weight => dl2/da  =weight.T
        
        return pre_dl_da

class Sequential:
    def __init__(self, layers: list):
        self.layers: list = layers
        self.layers[-1].is_output = True
    
    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        epsilon = 1e-9  # Small value to prevent division by zero
        return - (y_true / (y_pred + epsilon)) / y_true.shape[0]
    
    def fit(self, data, label, batch_size=10, epochs=2, learning_rate=None):
        if learning_rate is not None:
            for layer in self.layers:
                layer.learning_rate = learning_rate
                
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for i in range(0, data.shape[0], batch_size):
                data_batch = data[i:i+batch_size, :]      # shape: (batch_size, 784)
                label_batch = label[i:i+batch_size, :]    # shape: (batch_size, 10)
                
                # Forward pass
                A = data_batch
                for layer in self.layers:
                    A = layer.forward(A)

                # Calculate loss
                A_clipped = np.clip(A, 1e-9, 1.0) 
                loss = -np.sum(label_batch * np.log(A_clipped)) / label_batch.shape[0]
                total_loss += loss
                num_batches += 1
                
                # Backward pass
                dl_da = self.cross_entropy_derivative(label_batch, A)
                for layer in reversed(self.layers):
                    dl_da = layer.backward(dl_da)

            avg_loss = total_loss / num_batches
            print(f'========epoch: {epoch+1}||loss: {avg_loss:.4f}======================')
                
    def predict(self, test_data):
        A = test_data
        for layer in self.layers:
            A = layer.forward(A)

        return np.argmax(A, axis=1)  