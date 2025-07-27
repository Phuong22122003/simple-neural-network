class Layer:
    def forward(self, *args, **kwargs):
        raise NotImplementedError()
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError()
    
    def parameters(self):
        raise NotImplementedError()
    
    def __call__(self, *args, **kwds):
        self.forward(args, kwds)