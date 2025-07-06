class Layer:
    def __init__(self):
        pass
    def forward(self,inputs):
        raise NotImplementedError
    def backward(self,grad_outputs):
        raise NotImplementedError