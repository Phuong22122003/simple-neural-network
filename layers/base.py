class Layer:
    def forward(self, *args, **kwargs):
        raise NotImplementedError()
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError()
    
    def parameters(self):
        raise NotImplementedError()