class Dual:
    def __init__(self):
        self.forward = None
        self.backward = None
        self.weight_gradient = None
        self.weight_update = None
        pass

    def get_forward(self):
        return self.forward

    def get_backward(self):
        return self.backward

    def get_dual(self):
        return self.forward,self.backward

    @classmethod
    def from_torch_module(cls,module):
        assert False,"This function should be implemented."