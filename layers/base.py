class Layer:

    def forward(self,inputs):
        raise NotImplementedError()
    def backward(self,grad_outputs):
        raise NotImplementedError()
    def parameters(self):
        raise NotImplementedError()
    def update_weights(self,new_weights,new_bias):
        raise NotADirectoryError()
        